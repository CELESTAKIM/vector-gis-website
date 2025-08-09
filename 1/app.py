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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# DB config
DB_USER = 'postgres'
DB_PASS = 'KIM7222'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'gis_projects'

DB_URL = f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
engine = create_engine(DB_URL)

# Define the target projection for all data to ensure consistency.
# Leaflet uses Web Mercator (EPSG:3857) for its base tiles, but GeoJSON
# features are expected to be in WGS 84 (EPSG:4326). We'll use 4326.
TARGET_CRS = 'EPSG:4326'

# Helper: get list of spatial tables and metadata
def get_spatial_tables():
    """Returns dict of available PostGIS tables with basic info"""
    conn = psycopg2.connect(user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT, database=DB_NAME)
    cur = conn.cursor()
    # Only geometry tables with geom column (simplified)
    # We query the geometry_columns table which is maintained by PostGIS
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
            'type': 'point' if geom_type.lower() == 'point' else 'polygon'
        }
    cur.close()
    conn.close()
    return tables

def assign_color(table_name):
    """Simple color assigner by table name hash"""
    import hashlib
    # A list of distinct, eye-catching colors for the map layers
    colors = ['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#46f0f0','#f032e6','#bcf60c','#fabebe']
    h = int(hashlib.md5(table_name.encode()).hexdigest(), 16)
    return colors[h % len(colors)]

# Route: Homepage
@app.route('/')
def index():
    layers = get_spatial_tables()
    return render_template('index.html', layers=layers)

# Route: Return GeoJSON of a given table
@app.route('/data/<table>')
def get_layer_geojson(table):
    tables = get_spatial_tables()
    if table not in tables:
        return jsonify({'error': 'Layer not found'}), 404

    geom_col = tables[table]['geom_col']
    try:
        conn = psycopg2.connect(user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT, database=DB_NAME)
        cur = conn.cursor()
        # The query uses ST_AsGeoJSON to convert PostGIS geometry to GeoJSON format.
        # It also builds a JSON object for each feature, including a properties object.
        query = sql.SQL("""
            SELECT jsonb_build_object(
                'type', 'FeatureCollection',
                'features', jsonb_agg(feature)
            )
            FROM (
                SELECT jsonb_build_object(
                    'type', 'Feature',
                    'geometry', ST_AsGeoJSON(ST_Transform({geom}, 4326))::jsonb,
                    'properties', to_jsonb(t) - {geom}
                ) AS feature
                FROM {table} t
                LIMIT 1000
            ) features;
        """).format(
            geom=sql.Identifier(geom_col),
            table=sql.Identifier(tables[table]['schema'], table)
        )
        cur.execute(query)
        result = cur.fetchone()[0]
        cur.close()
        conn.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route: Return attribute table JSON for DataTables
@app.route('/attributes/<table>')
def get_attributes(table):
    tables = get_spatial_tables()
    if table not in tables:
        return jsonify({'error': 'Layer not found'}), 404
    geom_col = tables[table]['geom_col']
    try:
        # Use pandas to read the entire table into a DataFrame for attribute display
        sql_query = f"SELECT * FROM {tables[table]['schema']}.{table} LIMIT 1000"
        df = pd.read_sql(sql_query, con=engine)
        if geom_col in df.columns:
            # Drop the geometry column for the attribute table view
            df = df.drop(columns=[geom_col])
        df.fillna('', inplace=True)

        columns = list(df.columns)
        rows = df.to_dict(orient='records')

        # Add a synthetic rowid for client-side selection and highlighting
        for i, row in enumerate(rows):
            row['_rowid'] = i

        return jsonify({'columns': columns, 'rows': rows})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route: Upload shapefile (zip) and import into PostGIS
@app.route('/upload', methods=['POST'])
def upload_shapefile():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Sanitize new table name from form
    new_table = request.form.get('tablename', '').strip().lower()
    if not new_table.isidentifier():
        return jsonify({'error': 'Invalid table name. Use only letters, digits, and underscores.'}), 400

    # Check if table exists
    existing_tables = get_spatial_tables()
    if new_table in existing_tables:
        return jsonify({'error': 'Table already exists. Choose another name.'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, filename)
    file.save(zip_path)

    # Unzip shapefile contents
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({'error': f'Invalid zip file: {str(e)}'}), 400

    # Find the .shp file
    shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
    if not shp_files:
        shutil.rmtree(temp_dir)
        return jsonify({'error': 'No .shp file found in archive.'}), 400

    shp_path = os.path.join(temp_dir, shp_files[0])

    # Load shapefile with geopandas
    try:
        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            shutil.rmtree(temp_dir)
            return jsonify({'error': 'Shapefile contains no features.'}), 400
        
        # --- CRITICAL FIX: Reproject to the target CRS before saving ---
        if gdf.crs is None:
            shutil.rmtree(temp_dir)
            return jsonify({'error': 'Shapefile CRS is unknown. Cannot reproject.'}), 400
            
        gdf = gdf.to_crs(TARGET_CRS)
        
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({'error': f'Error reading shapefile or reprojecting: {str(e)}'}), 400

    # Save to PostGIS
    try:
        # `if_exists='fail'` prevents overwriting existing tables
        gdf.to_postgis(new_table, engine, if_exists='fail', index=False)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return jsonify({'error': f'Failed to save to PostGIS: {str(e)}'}), 500

    shutil.rmtree(temp_dir)
    return jsonify({'success': f'Layer "{new_table}" uploaded successfully!'})

# Route: Download selected features from a table as GeoJSON zipped
@app.route('/download', methods=['POST'])
def download_selected():
    data = request.get_json()
    table = data.get('layer')
    selected = data.get('selected', [])

    tables = get_spatial_tables()
    if table not in tables:
        return jsonify({'error': 'Layer not found'}), 404
    if not selected:
        return jsonify({'error': 'No features selected'}), 400

    geom_col = tables[table]['geom_col']

    # Get primary key or use ctid. This part of the code is simplified
    # since we are filtering by a row index from the frontend.
    try:
        df = pd.read_sql(f"SELECT * FROM {tables[table]['schema']}.{table}", con=engine)
        geom = tables[table]['geom_col']
        if geom in df.columns:
            geom_colname = geom
        else:
            geom_colname = None

        # Filter by row index (`_rowid` from the DataTables)
        filtered = df.iloc[selected]
        if geom_colname:
            # Create a GeoDataFrame with the correct CRS
            gdf = gpd.GeoDataFrame(filtered, geometry=geom_colname, crs=TARGET_CRS)
        else:
            gdf = gpd.GeoDataFrame(filtered)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch selected features: {str(e)}'}), 500

    # Export to GeoJSON and ZIP
    try:
        tmp_dir = tempfile.mkdtemp()
        geojson_path = os.path.join(tmp_dir, f"{table}_selection.geojson")
        # Ensure the output GeoJSON also has the correct CRS specified
        gdf.to_file(geojson_path, driver='GeoJSON', crs=TARGET_CRS)

        zip_path = os.path.join(tmp_dir, f"{table}_selection.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(geojson_path, arcname=f"{table}_selection.geojson")

        # Send file and cleanup later
        return send_file(zip_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Failed to create zip: {str(e)}'}), 500

# Route: Merge selected features from multiple tables
@app.route('/merge_layers', methods=['POST'])
def merge_layers():
    data = request.get_json()
    layers = data.get('layers', [])  # list of {layer: table_name, selected: [indexes]}
    if not layers:
        return jsonify({'error': 'No layers selected'}), 400

    try:
        merged_gdfs = []
        for layer_info in layers:
            table = layer_info.get('layer')
            selected = layer_info.get('selected', [])
            if not table or not selected:
                continue
            tables = get_spatial_tables()
            if table not in tables:
                continue
            geom_col = tables[table]['geom_col']
            df = pd.read_sql(f"SELECT * FROM {tables[table]['schema']}.{table}", con=engine)
            
            if geom_col in df.columns:
                # Ensure each GeoDataFrame is created with the target CRS
                gdf = gpd.GeoDataFrame(df, geometry=geom_col, crs=TARGET_CRS)
            else:
                gdf = gpd.GeoDataFrame(df)
            filtered = gdf.iloc[selected]
            merged_gdfs.append(filtered)

        if not merged_gdfs:
            return jsonify({'error': 'No valid selected features'}), 400

        result_gdf = gpd.GeoDataFrame(pd.concat(merged_gdfs, ignore_index=True))
        
        # --- CRITICAL FIX: Ensure the merged result has the correct CRS ---
        if result_gdf.crs is None:
            result_gdf.crs = TARGET_CRS
        else:
            result_gdf = result_gdf.to_crs(TARGET_CRS)
        
        tmp_dir = tempfile.mkdtemp()
        geojson_path = os.path.join(tmp_dir, f"merged_selection.geojson")
        result_gdf.to_file(geojson_path, driver='GeoJSON')

        zip_path = os.path.join(tmp_dir, f"merged_selection.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(geojson_path, arcname="merged_selection.geojson")

        return send_file(zip_path, as_attachment=True)
    except Exception as e:
        return jsonify({'error': f'Failed to merge layers: {str(e)}'}), 500

# Route: Delete a layer from the database
@app.route('/delete_layer/<layer_name>', methods=['DELETE'])
def delete_layer(layer_name):
    tables = get_spatial_tables()
    if layer_name not in tables:
        return jsonify({'error': 'Layer not found'}), 404
    
    try:
        conn = psycopg2.connect(user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT, database=DB_NAME)
        cur = conn.cursor()
        query = sql.SQL("DROP TABLE {} CASCADE").format(
            sql.Identifier(tables[layer_name]['schema'], layer_name)
        )
        cur.execute(query)
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'success': f'Layer "{layer_name}" deleted successfully.'})
    except Exception as e:
        return jsonify({'error': f'Failed to delete layer: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
