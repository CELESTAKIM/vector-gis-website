import os
import zipfile
import tempfile
import json
import shutil
import subprocess
import io
import base64
from pathlib import Path
from datetime import datetime

import psycopg2
from psycopg2 import sql
import geopandas as gpd
from sqlalchemy import create_engine
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from flask import Flask, render_template, jsonify, request, send_file
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database connection details
# NOTE: Ensure these credentials and database names are correct for your setup.
VECTOR_DB_CONFIG = {
    "user": "postgres",
    "password": "KIM7222",
    "host": "localhost",
    "port": "5432",
    "dbname": "gis_projects"
}
RASTER_DB_CONFIG = {
    "host": "localhost",
    "dbname": "DEM",
    "user": "postgres",
    "password": "KIM7222",
    "port": "5432"
}

VECTOR_DB_URL = f"postgresql://{VECTOR_DB_CONFIG['user']}:{VECTOR_DB_CONFIG['password']}@{VECTOR_DB_CONFIG['host']}:{VECTOR_DB_CONFIG['port']}/{VECTOR_DB_CONFIG['dbname']}"
engine = create_engine(VECTOR_DB_URL)

# --- Database Helper Functions ---
def get_vector_connection():
    """Establishes a connection to the vector database."""
    return psycopg2.connect(**VECTOR_DB_CONFIG)

def get_raster_connection():
    """Establishes a connection to the raster database."""
    return psycopg2.connect(**RASTER_DB_CONFIG)

def get_vector_tables():
    """Dynamically fetches all vector tables from the gis_projects database."""
    conn = get_vector_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT f_table_schema, f_table_name, f_geometry_column, type
        FROM geometry_columns
        WHERE f_table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY f_table_name;
    """)
    tables = {}
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
    i = 0
    for schema, table, geom_col, geom_type in cur.fetchall():
        tables[table] = {
            'schema': schema,
            'geom_col': geom_col,
            'title': table.replace('_', ' ').title(),
            'color': colors[i % len(colors)],
            'type': 'point' if 'point' in geom_type.lower() else 'polygon' if 'polygon' in geom_type.lower() else 'line',
            'db': 'vector'
        }
        i += 1
    cur.close()
    conn.close()
    return tables

def get_raster_tables():
    """Dynamically fetches all raster tables from the DEM database."""
    conn = get_raster_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT r_table_name
        FROM raster_columns
        WHERE r_table_schema = 'public'
        ORDER BY r_table_name;
    """)
    tables = {}
    for row in cur.fetchall():
        table_name = row[0]
        tables[table_name] = {
            'title': table_name.replace('_', ' ').title(),
            'type': 'raster',
            'db': 'raster'
        }
    cur.close()
    conn.close()
    return tables

def get_raster_column_name(table_name):
    """Fetches the name of the raster column for a given table."""
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

def get_all_layers():
    """Combines vector and raster tables into a single dictionary."""
    layers = get_vector_tables()
    layers.update(get_raster_tables())
    return layers

# --- Flask Routes ---
@app.route("/")
def index():
    """Renders the main page with a list of all available layers."""
    layers = get_all_layers()
    return render_template("index3.html", layers=layers)

# --- Vector Routes ---
@app.route('/data/<table>')
def get_layer_geojson(table):
    """Serves GeoJSON for a specific vector layer."""
    tables = get_vector_tables()
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
                    'properties', to_jsonb(t.*) - {geom_str} || jsonb_build_object('_rowid', row_number() OVER ())
                ) AS feature
                FROM {table} t
                LIMIT 10000
            ) features;
        """).format(
            geom=sql.Identifier(geom_col),
            geom_str=sql.Literal(geom_col),
            table=sql.Identifier(tables[table]['schema'], table)
        )
        cur.execute(query)
        result = cur.fetchone()
        geojson = result[0] if result and result[0] else {"type": "FeatureCollection", "features": []}
        cur.close()
        conn.close()
        return jsonify(geojson)
    except Exception as e:
        return jsonify({'error': f'Error fetching GeoJSON for {table}: {str(e)}'}), 500

@app.route('/attributes/<table>')
def get_attributes(table):
    """Returns the attribute table (columns and rows) for a vector layer."""
    tables = get_vector_tables()
    if table not in tables:
        return jsonify({'error': f'Layer {table} not found'}), 404
    
    geom_col = tables[table]['geom_col']
    try:
        query = f"SELECT *, row_number() OVER () AS _rowid FROM {tables[table]['schema']}.{table} LIMIT 10000"
        df = pd.read_sql(query, con=engine, index_col=None, coerce_float=True)
        
        # Remove geometry column from the attributes list
        columns = [col for col in df.columns if col not in [geom_col, 'geometry']]
        
        df.fillna('', inplace=True)
        rows = df.to_dict(orient='records')
        
        return jsonify({'columns': columns, 'rows': rows})
    except Exception as e:
        return jsonify({'error': f'Error fetching attributes for {table}: {str(e)}'}), 500

@app.route('/download', methods=['POST'])
def download_selected():
    """Downloads a zipped GeoJSON file of selected vector features."""
    data = request.get_json()
    table = data.get('layer')
    selected = data.get('selected', [])

    tables = get_vector_tables()
    if table not in tables:
        return jsonify({'error': f'Layer {table} not found'}), 404

    geom_col = tables[table]['geom_col']
    tmp_dir = None
    try:
        # Use row_number() to match the frontend selection
        query = f"""
            WITH numbered_rows AS (SELECT *, row_number() OVER () AS _rowid FROM {tables[table]['schema']}.{table})
            SELECT * FROM numbered_rows WHERE _rowid IN ({','.join(map(str, selected))})
        """
        
        gdf = gpd.read_postgis(query, con=engine, geom_col=geom_col)
        
        if gdf.empty:
            return jsonify({'error': 'No features selected for download.'}), 400

        tmp_dir = tempfile.mkdtemp()
        geojson_path = os.path.join(tmp_dir, f"{table}_selection.geojson")
        gdf.to_file(geojson_path, driver='GeoJSON')

        zip_path = os.path.join(tmp_dir, f"{table}_selection.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(geojson_path, arcname=f"{table}_selection.geojson")

        return send_file(zip_path, as_attachment=True, download_name=f"{table}_selection.zip")
    except Exception as e:
        return jsonify({'error': f'Failed to fetch selected features: {str(e)}'}), 500
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)

# --- Raster Routes ---
@app.route("/raster_tile/<table>/<int:z>/<int:x>/<int:y>.png")
def raster_tile(table, z, x, y):
    """Serves a raster tile for a given layer based on ZXY coordinates."""
    conn = None
    cur = None
    try:
        conn = get_raster_connection()
        cur = conn.cursor()
        
        raster_column = get_raster_column_name(table)
        if not raster_column:
            return jsonify({"error": "Invalid raster layer"}), 404
        
        query = sql.SQL("""
            SELECT ST_AsPNG(ST_Tile(rast, 1, 256, 256, true))
            FROM public.{}
            WHERE ST_Intersects(rast, ST_Transform(ST_TileEnvelope({}, {}, {}, {}), ST_SRID(rast)));
        """).format(
            sql.Identifier(table),
            sql.Literal(z), sql.Literal(x), sql.Literal(y), sql.Literal(4326)
        )
        cur.execute(query)
        
        tile_data = cur.fetchone()
        
        if tile_data and tile_data[0]:
            return send_file(io.BytesIO(tile_data[0]), mimetype='image/png')
        else:
            return send_file(io.BytesIO(b''), mimetype='image/png')
    except Exception as e:
        print(f"Error generating raster tile: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if cur: cur.close()
        if conn: conn.close()

@app.route("/raster_download/<table>")
def raster_download(table):
    """Downloads the full GeoTIFF raster data for a given layer."""
    try:
        conn = get_raster_connection()
        cur = conn.cursor()
        raster_column = get_raster_column_name(table)
        
        if not raster_column:
            return jsonify({"error": f"Raster column not found for table {table}"}), 404

        cur.execute(f"""
            SELECT ST_AsGDALRaster(ST_Union({raster_column}), 'GTiff')
            FROM public.{table};
        """)
        
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
            download_name=f'{table}.tif'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- File Upload Routes ---
@app.route('/upload_vector', methods=['POST'])
def upload_vector():
    """Uploads a zipped shapefile and imports it into the vector database."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file part or empty filename'}), 400
    
    file = request.files['file']
    new_table = request.form.get('tablename', '').strip().lower()
    
    if not new_table.isidentifier():
        return jsonify({'error': 'Invalid table name. Use only letters, digits, and underscores.'}), 400
    
    existing_tables = get_vector_tables()
    if new_table in existing_tables:
        return jsonify({'error': 'Table already exists. Choose another name.'}), 400

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, secure_filename(file.filename))
    file.save(zip_path)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
        if not shp_files:
            return jsonify({'error': 'No .shp file found in archive.'}), 400
        
        shp_path = os.path.join(temp_dir, shp_files[0])
        gdf = gpd.read_file(shp_path)
        gdf.to_postgis(new_table, engine, if_exists='fail', index=False, schema='public')
        return jsonify({'success': f'Vector layer "{new_table}" uploaded successfully!'})
    except Exception as e:
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 500
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/upload_raster', methods=['POST'])
def upload_raster():
    """Uploads a zipped GeoTIFF and imports it into the raster database."""
    if 'file' not in request.files or not request.files['file'].filename:
        return jsonify({'error': 'No file part or empty filename'}), 400
    
    file = request.files['file']
    new_table = request.form.get('tablename', '').strip().lower()

    if not new_table.isidentifier():
        return jsonify({'error': 'Invalid table name. Use only letters, digits, and underscores.'}), 400
    
    conn = get_raster_connection()
    cur = conn.cursor()
    cur.execute("SELECT r_table_name FROM raster_columns WHERE r_table_schema = 'public'")
    existing_tables = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    
    if new_table in existing_tables:
        return jsonify({'error': 'Table already exists. Choose another name.'}), 400

    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, secure_filename(file.filename))
    file.save(zip_path)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        tif_files = [f for f in os.listdir(temp_dir) if f.endswith(('.tif', '.tiff'))]
        if not tif_files:
            return jsonify({'error': 'No .tif or .tiff file found in archive.'}), 400
        
        tif_path = os.path.join(temp_dir, tif_files[0])
        with rasterio.open(tif_path) as src:
            srid = src.crs.to_epsg() if src.crs else 4326

        cmd = [
            'raster2pgsql', '-I', '-C', '-s', str(srid), '-F', '-t', '100x100',
            str(tif_path), f'public.{new_table}'
        ]
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sql_output = process.stdout.decode()
        
        conn = get_raster_connection()
        with conn.cursor() as cur:
            cur.execute(sql_output)
        conn.commit()
        return jsonify({'success': f'Raster layer "{new_table}" uploaded successfully!'})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Failed to import raster to PostGIS: {e.stderr.decode()}'}), 500
    except Exception as e:
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 500
    finally:
        if 'conn' in locals() and not conn.closed:
            conn.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

# --- Layer Management Routes ---
@app.route('/delete_layer/vector/<table>', methods=['POST'])
def delete_vector_layer(table):
    """Deletes a vector layer from the gis_projects database."""
    tables = get_vector_tables()
    if table not in tables:
        return jsonify({'error': f'Layer {table} not found'}), 404
    
    try:
        conn = get_vector_connection()
        cur = conn.cursor()
        query = sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
            sql.Identifier(tables[table]['schema']), sql.Identifier(table)
        )
        cur.execute(query)
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'success': f'Vector layer "{table}" deleted successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_layer/raster/<table>', methods=['POST'])
def delete_raster_layer(table):
    """Deletes a raster layer from the DEM database."""
    try:
        conn = get_raster_connection()
        cur = conn.cursor()
        query = sql.SQL("DROP TABLE IF EXISTS public.{}").format(sql.Identifier(table))
        cur.execute(query)
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'success': f'Raster layer "{table}" deleted successfully!'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/create_layer', methods=['POST'])
def create_layer_from_selection():
    """Creates a new vector layer from selected features."""
    data = request.get_json()
    source_table = data.get('source_table')
    new_table = data.get('new_table')
    rowids = data.get('rowids', [])

    tables = get_vector_tables()
    if source_table not in tables:
        return jsonify({'error': f'Source layer {source_table} not found'}), 404
    if not new_table.isidentifier():
        return jsonify({'error': 'Invalid new table name.'}), 400
    if new_table in tables:
        return jsonify({'error': 'New table name already exists.'}), 400
    if not rowids:
        return jsonify({'error': 'No features selected to create layer.'}), 400

    try:
        conn = get_vector_connection()
        cur = conn.cursor()
        # Create a new table with selected features
        query_create = sql.SQL("""
            CREATE TABLE public.{} AS
            WITH numbered_rows AS (SELECT *, row_number() OVER () AS _rowid FROM {}.{})
            SELECT * FROM numbered_rows WHERE _rowid IN %s;
        """).format(
            sql.Identifier(new_table),
            sql.Identifier(tables[source_table]['schema']),
            sql.Identifier(source_table)
        )
        cur.execute(query_create, (tuple(rowids),))
        conn.commit()
        
        # Add a geometry index to the new table
        geom_col = tables[source_table]['geom_col']
        query_index = sql.SQL("CREATE INDEX ON public.{} USING GIST({});").format(
            sql.Identifier(new_table),
            sql.Identifier(geom_col)
        )
        cur.execute(query_index)
        conn.commit()
        
        return jsonify({'success': f'New layer "{new_table}" created successfully!'})
    except Exception as e:
        return jsonify({'error': f'Failed to create layer: {str(e)}'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
