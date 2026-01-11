module.exports = {
  root: true,
  extends: ["eslint:recommended", "plugin:astro/recommended"],
  parser: "astro-eslint-parser",
  parserOptions: {
    parser: "@typescript-eslint/parser",
    extraFileExtensions: [".astro"],
  },
  env: {
    browser: true,
    es2021: true,
    node: true,
  },
};
