import os
import zipfile
import tempfile
import json
import shutil
from flask import Flask, render_template, jsonify, request, send_file
from werkzeug.utils import secure_filename
import psycopg2
from psycopg2 import sql
import geopandas as gpd
from shapely.geometry import mapping
from sqlalchemy import create_engine
import pandas as pd
import io
import rasterio
from rasterio.warp import transform_bounds
import subprocess

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Database Configurations ---
# Credentials provided by Kimathi Joram (celestakim018@gmail.com), GIS & RS enthusiast at DeKUT.
RASTER_DB_CONFIG = {
    "host": "localhost",
    "dbname": "DEM",
    "user": "postgres",
    "password": "KIM7222",
    "port": "5432"
}

VECTOR_DB_CONFIG = {
    "user": "postgres",
    "password": "KIM7222",
    "host": "localhost",
    "port": "5432",
    "dbname": "gis_projects"
}

VECTOR_DB_URL = f"postgresql://{VECTOR_DB_CONFIG['user']}:{VECTOR_DB_CONFIG['password']}@{VECTOR_DB_CONFIG['host']}:{VECTOR_DB_CONFIG['port']}/{VECTOR_DB_CONFIG['dbname']}"
engine = create_engine(VECTOR_DB_URL)

# --- Database Connection Functions ---
def get_raster_connection():
    return psycopg2.connect(**RASTER_DB_CONFIG)

def get_vector_connection():
    return psycopg2.connect(**VECTOR_DB_CONFIG)

def get_raster_column_name(table_name):
    conn = None
    cur = None
    try:
        conn = get_raster_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT r_raster_column
            FROM raster_columns
            WHERE r_table_name = %s AND r_table_schema = 'public'
            LIMIT 1;
        """, (table_name.lower(),))
        result = cur.fetchone()
        return result[0] if result else None
    except Exception as e:
        print(f"Error fetching raster column name for {table_name}: {e}")
        return None
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def get_spatial_tables():
    conn = get_vector_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT f_table_schema, f_table_name, f_geometry_column, type 
        FROM geometry_columns
        WHERE f_table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY f_table_name;
    """)
    tables = {}
    for schema, table, geom_col, geom_type in cur.fetchall():
        tables[table] = {
            'schema': schema,
            'geom_col': geom_col,
            'geom_type': geom_type,
            'title': table.replace('_', ' ').title(),
            'color': assign_color(table),
            'type': 'point' if geom_type.lower() == 'point' else 'polygon' if geom_type.lower() in ('polygon', 'multipolygon') else 'line'
        }
    cur.close()
    conn.close()
    return tables

def assign_color(table_name):
    import hashlib
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
    h = int(hashlib.md5(table_name.encode()).hexdigest(), 16)
    return colors[h % len(colors)]

def create_styles_table():
    conn = get_vector_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS layer_styles (
            table_name VARCHAR PRIMARY KEY,
            color VARCHAR(7)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

create_styles_table()

# --- Flask Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/spatial_tables")
def spatial_tables():
    try:
        return jsonify(get_spatial_tables())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Raster Routes ---

@app.route("/raster_tables")
def raster_tables():
    try:
        conn = get_raster_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT r_table_name
            FROM raster_columns
            WHERE r_table_schema = 'public'
            ORDER BY r_table_name;
        """)
        tables = [row[0] for row in cur.fetchall()]

        all_data = []
        for table in tables:
            raster_column = get_raster_column_name(table)
            if not raster_column:
                continue
            cur.execute(f"""
                SELECT rid,
                       ST_YMin(ST_Envelope({raster_column})) AS ymin,
                       ST_XMin(ST_Envelope({raster_column})) AS xmin,
                       ST_YMax(ST_Envelope({raster_column})) AS ymax,
                       ST_XMax(ST_Envelope({raster_column})) AS xmax
                FROM public.{table}
                ORDER BY rid;
            """)
            rasters = []
            for rid, ymin, xmin, ymax, xmax in cur.fetchall():
                rasters.append({
                    "rid": rid,
                    "bounds": [[ymin, xmin], [ymax, xmax]]
                })
            all_data.append({
                "table_name": table,
                "rasters": rasters
            })

        cur.close()
        conn.close()
        return jsonify(all_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/raster_preview/<table>/<int:rid>")
def raster_preview(table, rid):
    try:
        conn = get_raster_connection()
        cur = conn.cursor()
        raster_column = get_raster_column_name(table)
        if not raster_column:
            return jsonify({"error": f"Raster column not found for table {table}"}), 404

        cur.execute(f"""
            SELECT (stats).min, (stats).max
            FROM (
                SELECT ST_SummaryStats({raster_column}, 1, TRUE) AS stats
                FROM public.{table}
                WHERE rid = %s
            ) AS foo;
        """, (rid,))
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "No stats found"}), 404
        min_val, max_val = row

        cur.execute(f"""
            SELECT encode(
                ST_AsPNG(
                    ST_Reclass(
                        {raster_column}, 1,
                        %s,
                        '8BUI', 0
                    )
                ), 'base64'),
                ST_YMin(ST_Envelope({raster_column})),
                ST_XMin(ST_Envelope({raster_column})),
                ST_YMax(ST_Envelope({raster_column})),
                ST_XMax(ST_Envelope({raster_column}))
            FROM public.{table}
            WHERE rid = %s;
        """, (f"{min_val}-{max_val}:0-255", rid))

        img_row = cur.fetchone()
        cur.close()
        conn.close()

        if not img_row or not img_row[0]:
            return jsonify({"error": "No image found"}), 404

        png_b64, ymin, xmin, ymax, xmax = img_row

        return jsonify({
            "image": png_b64,
            "bounds": [[ymin, xmin], [ymax, xmax]],
            "stats": {"min": min_val, "max": max_val}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/raster_download/<table>/<int:rid>")
def raster_download(table, rid):
    try:
        conn = get_raster_connection()
        cur = conn.cursor()
        raster_column = get_raster_column_name(table)
        if not raster_column:
            return jsonify({"error": f"Raster column not found for table {table}"}), 404

        cur.execute(f"""
            SELECT ST_AsGDALRaster({raster_column}, 'GTiff')
            FROM public.{table}
            WHERE rid = %s;
        """, (rid,))
        
        gdal_raster_data = cur.fetchone()
        cur.close()
        conn.close()
        
        if not gdal_raster_data:
            return jsonify({"error": "No raster data found for download."}), 404
            
        tiff_data = gdal_raster_data[0]
        tiff_file = io.BytesIO(tiff_data)
        tiff_file.seek(0)
        
        return send_file(
            tiff_file,
            mimetype='image/tiff',
            as_attachment=True,
            download_name=f'{table}_{rid}.tif'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download_raster/<table>')
def download_raster_layer(table):
    try:
        conn = get_raster_connection()
        cur = conn.cursor()
        raster_column = get_raster_column_name(table)
        if not raster_column:
            return jsonify({"error": f"Raster column not found for table {table}"}), 404

        cur.execute(f"""
            SELECT rid, ST_AsGDALRaster({raster_column}, 'GTiff')
            FROM public.{table}
            ORDER BY rid;
        """)
        rasters = cur.fetchall()
        cur.close()
        conn.close()

        if not rasters:
            return jsonify({"error": f"No raster data found in table {table}"}), 404

        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, f"{table}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for rid, raster_data in rasters:
                tiff_path = os.path.join(tmp_dir, f"{table}_{rid}.tif")
                with open(tiff_path, 'wb') as f:
                    f.write(raster_data)
                zipf.write(tiff_path, arcname=f"{table}_{rid}.tif")
                os.remove(tiff_path)

        return send_file(
            zip_path,
            as_attachment=True,
            mimetype='application/zip',
            download_name=f'{table}.zip'
        )
    except Exception as e:
        return jsonify({"error": f"Failed to download raster layer: {str(e)}"}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.route('/upload_raster', methods=['POST'])
def upload_raster():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    new_table = request.form.get('tablename', '').strip().lower()
    if not new_table.isidentifier():
        return jsonify({'error': 'Invalid table name. Use only letters, digits, and underscores.'}), 400

    conn = get_raster_connection()
    cur = conn.cursor()
    cur.execute("SELECT r_table_name FROM raster_columns WHERE r_table_schema = 'public'")
    existing_tables = [row[0] for row in cur.fetchall()]
    cur.close()
    if new_table in existing_tables:
        return jsonify({'error': 'Table already exists. Choose another name.'}), 400

    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, filename)
    file.save(zip_path)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({'error': f'Invalid zip file: {str(e)}'}), 400

    tif_files = [f for f in os.listdir(temp_dir) if f.endswith(('.tif', '.tiff'))]
    if not tif_files:
        shutil.rmtree(temp_dir)
        return jsonify({'error': 'No .tif or .tiff file found in archive.'}), 400

    tif_path = os.path.join(temp_dir, tif_files[0])

    try:
        with rasterio.open(tif_path) as src:
            if src.count == 0:
                shutil.rmtree(temp_dir)
                return jsonify({'error': 'Raster file contains no data.'}), 400
            crs = src.crs
            srid = 4326 if crs is None else src.crs.to_epsg()
            if srid is None:
                srid = 4326
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({'error': f'Error reading raster file: {str(e)}'}), 400

    try:
        conn = get_raster_connection()
        cur = conn.cursor()
        cur.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} (rid serial PRIMARY KEY, rast raster)").format(sql.Identifier('public', new_table)))
        conn.commit()

        cmd = [
            'raster2pgsql',
            '-s', str(srid),
            '-I', '-C', '-M',
            tif_path,
            f'public.{new_table}'
        ]
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            shutil.rmtree(temp_dir)
            return jsonify({'error': f'Failed to import raster to PostGIS: {process.stderr}'}), 500

        with open('raster_load.sql', 'w') as f:
            f.write(process.stdout)
        subprocess.run(['psql', '-d', RASTER_DB_CONFIG['dbname'], '-U', RASTER_DB_CONFIG['user'], '-f', 'raster_load.sql'])
        os.remove('raster_load.sql')
        conn.commit()
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({'error': f'Failed to save raster to PostGIS: {str(e)}'}), 500
    finally:
        if conn:
            conn.close()

    shutil.rmtree(temp_dir)
    return jsonify({'success': f'Raster layer "{new_table}" uploaded successfully!'})

# --- Vector Routes ---

@app.route('/data/<table>')
def get_layer_geojson(table):
    tables = get_spatial_tables()
    if table not in tables:
        return jsonify({'error': f'Layer {table} not found'}), 404

    geom_col = tables[table]['geom_col']
    try:
        conn = get_vector_connection()
        cur = conn.cursor()
        query = sql.SQL("""
            SELECT jsonb_build_object(
                'type', 'FeatureCollection',
                'features', jsonb_agg(feature)
            )
            FROM (
                SELECT jsonb_build_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON({geom})::jsonb,
                    'properties', to_jsonb(t.*) - {geom} || jsonb_build_object('_rowid', row_number() OVER ())
                ) AS feature
                FROM {table} t
                LIMIT 1000
            ) features;
        """).format(
            geom=sql.Identifier(geom_col),
            table=sql.Identifier(tables[table]['schema'], table)
        )
        cur.execute(query)
        result = cur.fetchone()
        if not result or not result[0]:
            return jsonify({'error': f'No features found in {table}'}), 404
        geojson = result[0]
        cur.close()
        conn.close()
        return jsonify(geojson)
    except Exception as e:
        return jsonify({'error': f'Error fetching GeoJSON for {table}: {str(e)}'}), 500

@app.route('/attributes/<table>')
def get_attributes(table):
    tables = get_spatial_tables()
    if table not in tables:
        return jsonify({'error': f'Layer {table} not found'}), 404
    geom_col = tables[table]['geom_col']
    try:
        sql_query = f"SELECT * FROM {tables[table]['schema']}.{table} LIMIT 1000"
        df = pd.read_sql(sql_query, con=engine)
        if geom_col in df.columns:
            df = df.drop(columns=[geom_col])
        df.fillna('', inplace=True)

        columns = list(df.columns)
        rows = df.to_dict(orient='records')

        for i, row in enumerate(rows):
            row['_rowid'] = i

        return jsonify({'columns': columns, 'rows': rows})
    except Exception as e:
        return jsonify({'error': f'Error fetching attributes for {table}: {str(e)}'}), 500

@app.route('/upload_vector', methods=['POST'])
def upload_vector():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    new_table = request.form.get('tablename', '').strip().lower()
    if not new_table.isidentifier():
        return jsonify({'error': 'Invalid table name. Use only letters, digits, and underscores.'}), 400

    existing_tables = get_spatial_tables()
    if new_table in existing_tables:
        return jsonify({'error': 'Table already exists. Choose another name.'}), 400

    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, filename)
    file.save(zip_path)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({'error': f'Invalid zip file: {str(e)}'}), 400

    shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
    if not shp_files:
        shutil.rmtree(temp_dir)
        return jsonify({'error': 'No .shp file found in archive.'}), 400

    shp_path = os.path.join(temp_dir, shp_files[0])

    try:
        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            shutil.rmtree(temp_dir)
            return jsonify({'error': 'Shapefile contains no features.'}), 400
        gdf['row_number'] = range(len(gdf))
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({'error': f'Error reading shapefile: {str(e)}'}), 400

    try:
        gdf.to_postgis(new_table, engine, if_exists='fail', index=False, schema='public')
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({'error': f'Failed to save to PostGIS: {str(e)}'}), 500

    shutil.rmtree(temp_dir)
    return jsonify({'success': f'Vector layer "{new_table}" uploaded successfully!'})

@app.route('/download', methods=['POST'])
def download_selected():
    data = request.get_json()
    table = data.get('layer')
    selected = data.get('selected', [])

    tables = get_spatial_tables()
    if table not in tables:
        return jsonify({'error': f'Layer {table} not found'}), 404

    geom_col = tables[table]['geom_col']
    try:
        conn = get_vector_connection()
        cur = conn.cursor()
        cur.execute(sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s AND column_name NOT IN (%s, %s) LIMIT 1"),
                    (tables[table]['schema'], table, geom_col, 'geom'))
        pk_col = cur.fetchone()
        pk_col = pk_col[0] if pk_col else 'ctid'
        cur.close()
        conn.close()
    except:
        pk_col = 'ctid'

    try:
        query = f"SELECT * FROM {tables[table]['schema']}.{table}"
        if selected:
            query += f" WHERE row_number IN ({','.join(map(str, selected))})"
        df = pd.read_sql(query, con=engine)
        geom = tables[table]['geom_col']
        geom_colname = geom if geom in df.columns else None

        if geom_colname:
            gdf = gpd.GeoDataFrame(df, geometry=geom_colname, crs='EPSG:4326')
        else:
            gdf = gpd.GeoDataFrame(df)

        tmp_dir = tempfile.mkdtemp()
        geojson_path = os.path.join(tmp_dir, f"{table}_selection.geojson")
        gdf.to_file(geojson_path, driver='GeoJSON')

        zip_path = os.path.join(tmp_dir, f"{table}_selection.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(geojson_path, arcname=f"{table}_selection.geojson")

        return send_file(zip_path, as_attachment=True, mimetype='application/zip')
    except Exception as e:
        return jsonify({'error': f'Failed to fetch selected features: {str(e)}'}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.route('/download_layer/<table>')
def download_layer(table):
    tables = get_spatial_tables()
    if table not in tables:
        return jsonify({'error': f'Layer {table} not found'}), 404
    try:
        query = f"SELECT * FROM {tables[table]['schema']}.{table}"
        gdf = gpd.read_postgis(query, con=engine, geom_col=tables[table]['geom_col'])
        tmp_dir = tempfile.mkdtemp()
        geojson_path = os.path.join(tmp_dir, f"{table}.geojson")
        gdf.to_file(geojson_path, driver='GeoJSON')
        zip_path = os.path.join(tmp_dir, f"{table}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(geojson_path, arcname=f"{table}.geojson")
        return send_file(zip_path, as_attachment=True, mimetype='application/zip')
    except Exception as e:
        return jsonify({'error': f'Failed to download layer: {str(e)}'}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.route('/merge_features', methods=['POST'])
def merge_features():
    data = request.get_json()
    table = data.get('table')
    rowids = data.get('rowids', [])
    new_table = data.get('new_table')

    tables = get_spatial_tables()
    if table not in tables:
        return jsonify({'error': f'Layer {table} not found'}), 404
    if not new_table or not new_table.isidentifier():
        return jsonify({'error': 'Invalid new table name.'}), 400
    if new_table in tables:
        return jsonify({'error': 'Table already exists.'}), 400
    if len(rowids) < 2:
        return jsonify({'error': 'Select at least two features to merge.'}), 400

    geom_col = tables[table]['geom_col']
    try:
        conn = get_vector_connection()
        cur = conn.cursor()
        cur.execute(sql.SQL("""
            SELECT COUNT(*) > 0
            FROM {table} a
            JOIN {table} b ON ST_Touches(a.{geom}, b.{geom})
            WHERE a.row_number IN %s AND b.row_number IN %s AND a.row_number < b.row_number
        """).format(
            table=sql.Identifier(tables[table]['schema'], table),
            geom=sql.Identifier(geom_col)
        ), (tuple(rowids), tuple(rowids)))
        if not cur.fetchone()[0]:
            return jsonify({'error': 'Selected features do not share boundaries.'}), 400

        cur.execute(sql.SQL("""
            CREATE TABLE {new_table} AS
            SELECT ST_Union({geom}) AS {geom}, 
                   MIN(row_number) AS row_number,
                   COUNT(*) AS merged_count
            FROM {source_table}
            WHERE row_number IN %s
            GROUP BY ST_Union({geom});
        """).format(
            new_table=sql.Identifier('public', new_table),
            source_table=sql.Identifier(tables[table]['schema'], table),
            geom=sql.Identifier(geom_col)
        ), (tuple(rowids),))
        conn.commit()

        cur.execute(sql.SQL("""
            SELECT UpdateGeometrySRID('public', {table}, {geom}, 4326);
        """).format(
            table=sql.Literal(new_table),
            geom=sql.Literal(geom_col)
        ))
        conn.commit()

        cur.close()
        conn.close()
        return jsonify({'success': f'New layer "{new_table}" created from merged features!'})
    except Exception as e:
        return jsonify({'error': f'Failed to merge features: {str(e)}'}), 500

@app.route('/merge_layers', methods=['POST'])
def merge_layers():
    data = request.get_json()
    layers = data.get('layers', [])
    if not layers:
        return jsonify({'error': 'No layers selected'}), 400

    try:
        merged_gdfs = []
        conn = get_vector_connection()
        cur = conn.cursor()
        for layer_info in layers:
            table = layer_info.get('layer')
            selected = layer_info.get('selected', [])
            if not table or not selected:
                continue
            tables = get_spatial_tables()
            if table not in tables:
                continue
            geom_col = tables[table]['geom_col']

            query = sql.SQL("""
                SELECT t1.*
                FROM {table} t1
                WHERE row_number IN %s
            """).format(
                table=sql.Identifier(tables[table]['schema'], table)
            )
            cur.execute(query, (tuple(selected),))
            rows = cur.fetchall()
            if rows:
                df = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])
                if geom_col in df.columns:
                    gdf = gpd.GeoDataFrame(df, geometry=geom_col, crs='EPSG:4326')
                    merged_gdfs.append(gdf)

        cur.close()
        conn.close()

        if not merged_gdfs:
            return jsonify({'error': 'No valid selected features'}), 400

        result_gdf = gpd.GeoDataFrame(pd.concat(merged_gdfs, ignore_index=True))
        tmp_dir = tempfile.mkdtemp()
        geojson_path = os.path.join(tmp_dir, "merged_selection.geojson")
        result_gdf.to_file(geojson_path, driver='GeoJSON')

        zip_path = os.path.join(tmp_dir, "merged_selection.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(geojson_path, arcname="merged_selection.geojson")

        return send_file(zip_path, as_attachment=True, mimetype='application/zip')
    except Exception as e:
        return jsonify({'error': f'Failed to merge layers: {str(e)}'}), 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.route('/delete/<table>', methods=['POST'])
def delete_layer(table):
    tables = get_spatial_tables()
    if table not in tables:
        return jsonify({'error': f'Layer {table} not found'}), 404

    try:
        conn = get_vector_connection()
        cur = conn.cursor()
        query = sql.SQL("DROP TABLE IF EXISTS {}").format(
            sql.Identifier(tables[table]['schema'], table)
        )
        cur.execute(query)
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'success': f'Layer "{table}" deleted successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/create_layer', methods=['POST'])
def create_layer():
    data = request.get_json()
    source_table = data.get('source_table')
    new_table = data.get('new_table')
    rowids = data.get('rowids', [])

    tables = get_spatial_tables()
    if source_table not in tables:
        return jsonify({'error': f'Source layer {source_table} not found'}), 404
    if not new_table or not new_table.isidentifier():
        return jsonify({'error': 'Invalid new table name. Use only letters, digits, and underscores.'}), 400
    if new_table in tables:
        return jsonify({'error': 'New table name already exists. Choose another name.'}), 400
    if not rowids:
        return jsonify({'error': 'No features selected to create layer.'}), 400

    geom_col = tables[source_table]['geom_col']
    try:
        conn = get_vector_connection()
        cur = conn.cursor()
        cur.execute(sql.SQL("""
            CREATE TABLE {new_table} AS
            SELECT *
            FROM {source_table}
            WHERE row_number IN %s;
        """).format(
            new_table=sql.Identifier('public', new_table),
            source_table=sql.Identifier(tables[source_table]['schema'], source_table)
        ), (tuple(rowids),))
        conn.commit()

        cur.execute(sql.SQL("""
            SELECT UpdateGeometrySRID('public', {table}, {geom}, 4326);
        """).format(
            table=sql.Literal(new_table),
            geom=sql.Literal(geom_col)
        ))
        conn.commit()

        cur.close()
        conn.close()
        return jsonify({'success': f'New layer "{new_table}" created successfully from selected features!'})
    except Exception as e:
        return jsonify({'error': f'Failed to create layer: {str(e)}'}), 500

@app.route('/set_color/<table>', methods=['POST'])
def set_color(table):
    tables = get_spatial_tables()
    if table not in tables:
        return jsonify({'error': f'Layer {table} not found'}), 404
    data = request.get_json()
    color = data.get('color')
    if not color or not color.startswith('#') or len(color) != 7:
        return jsonify({'error': 'Invalid color format. Use #RRGGBB.'}), 400
    try:
        conn = get_vector_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO layer_styles (table_name, color)
            VALUES (%s, %s)
            ON CONFLICT (table_name) DO UPDATE
            SET color = EXCLUDED.color;
        """, (table, color))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'success': f'Color updated for {table}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_color/<table>')
def get_color(table):
    try:
        conn = get_vector_connection()
        cur = conn.cursor()
        cur.execute("SELECT color FROM layer_styles WHERE table_name = %s", (table,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        return jsonify({'color': result[0] if result else assign_color(table)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
