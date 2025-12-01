// src/config/env.ts
import 'dotenv/config';

const required = (value: string | undefined, key: string): string => {
  if (!value) {
    throw new Error(`Missing env var: ${key}`);
  }
  return value;
};

export const env = {
  NODE_ENV: process.env.NODE_ENV ?? 'development',
  PORT: parseInt(process.env.PORT || '4000', 10),
  JWT_SECRET: required(process.env.JWT_SECRET, 'JWT_SECRET'),
  DATABASE_URL: required(process.env.DATABASE_URL, 'DATABASE_URL'),
};
