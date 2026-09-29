import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import * as topojson from 'topojson-server';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const geoPath = path.resolve(__dirname, '../src/data/india_subdivisions_geo.json');
const topoPath = path.resolve(__dirname, '../src/data/india_subdivisions_topo.json');

console.log('Reading GeoJSON from:', geoPath);
const geoData = JSON.parse(fs.readFileSync(geoPath, 'utf8'));

console.log(`Converting ${geoData.features.length} features to TopoJSON...`);
const topo = topojson.topology({ subdivisions: geoData }, 1e5);

fs.writeFileSync(topoPath, JSON.stringify(topo), 'utf8');
const stat = fs.statSync(topoPath);
console.log(`Successfully generated TopoJSON: ${topoPath} (${Math.round(stat.size / 1024)} KB)`);
