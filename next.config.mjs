import fs from 'node:fs';
import path from 'node:path';

const DEMO_DATA_MIN_ROWS = 100;
const DEMO_DATA_PATH = path.join(process.cwd(), 'public', 'demo-data.json');

const FIELD_RANGES = [
  ['rating', 1, 5],
  ['vader.compound', -1, 1],
  ['vader.pos', 0, 1],
  ['vader.neg', 0, 1],
  ['vader.neu', 0, 1],
  ['roberta.positive', 0, 1],
  ['roberta.neutral', 0, 1],
  ['roberta.negative', 0, 1],
  ['disagreement', 0, 2],
];

const failBuild = (message) => {
  throw new Error(`BUILD FAILED: ${message}`);
};

const getNestedValue = (value, fieldPath) => (
  fieldPath.split('.').reduce((current, key) => current?.[key], value)
);

const assertFiniteNumberInRange = (value, fieldPath, min, max, rowIndex) => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    failBuild(`public/demo-data.json row ${rowIndex} field "${fieldPath}" must be a finite number from ${min} to ${max}`);
  }
};

(() => {
  if (!fs.existsSync(DEMO_DATA_PATH)) {
    failBuild('public/demo-data.json does not exist');
  }

  let data;
  try {
    data = JSON.parse(fs.readFileSync(DEMO_DATA_PATH, 'utf8'));
  } catch (error) {
    failBuild(`public/demo-data.json could not be parsed as JSON: ${error.message}`);
  }

  if (!Array.isArray(data)) {
    failBuild('public/demo-data.json must contain a JSON array');
  }

  if (data.length < DEMO_DATA_MIN_ROWS) {
    failBuild(`public/demo-data.json must contain at least ${DEMO_DATA_MIN_ROWS} rows; found ${data.length}`);
  }

  data.forEach((row, rowIndex) => {
    if (typeof row !== 'object' || row === null || Array.isArray(row)) {
      failBuild(`public/demo-data.json row ${rowIndex} must be an object`);
    }

    if (typeof row.text !== 'string' || row.text.trim().length === 0) {
      failBuild(`public/demo-data.json row ${rowIndex} field "text" must be a non-empty string`);
    }

    FIELD_RANGES.forEach(([fieldPath, min, max]) => {
      assertFiniteNumberInRange(getNestedValue(row, fieldPath), fieldPath, min, max, rowIndex);
    });
  });

  console.log(`✓ demo-data.json validated: ${data.length} rows, schema OK`);
})();

/** @type {import('next').NextConfig} */
const nextConfig = {};

export default nextConfig;
