# Custom Agent Rules for Vault Workspace

- **TSConfig Integrity**: DO NOT add the deprecated `"baseUrl"` option back to `frontend/tsconfig.json` or `frontend/tsconfig.app.json`. This option was deprecated in TypeScript 6.0 (and will be removed in TypeScript 7.0). Using `"baseUrl"` causes compilation errors under TS 6.0+ unless silenced with `"ignoreDeprecations"`.
- **Path Resolution**: Keep the path resolution mapping in `tsconfig.app.json` and `tsconfig.json` without using `baseUrl`, which is fully supported under the `"moduleResolution": "bundler"` configuration.
- **Rule Persistence**: Always preserve the cleaned `tsconfig` files as-is.
