import react from '@vitejs/plugin-react-swc';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';

export default defineConfig(({mode}) => {
    loadEnv(mode, process.cwd(), '');
    return {
        // base url for the app - remove the specific path since this is a standalone translation app
        base: '/',

        // include .txt files as assets (useful for language data files)
        assetsInclude: ['**/*.txt'],

        // optimize build for translation app
        build: {
            rollupOptions: {
                output: {
                    manualChunks(id) {
                        // split out MUI components since it's a large UI library
                        if (id.includes('node_modules/@mui')) return 'vendor-mui';

                        // isolate emotion (used by MUI) for better caching and load performance
                        if (id.includes('node_modules/@emotion')) return 'vendor-emotion';

                        // group React and related packages
                        if (id.includes('node_modules/react') || id.includes('node_modules/react-dom')) {
                            return 'vendor-react';
                        }

                        // group utility packages
                        if (id.includes('node_modules/lodash') || id.includes('node_modules/date-fns') || id.includes('node_modules/uuid')) {
                            return 'vendor-utils';
                        }

                        // fallback chunk for any remaining node_modules
                        if (id.includes('node_modules')) return 'vendor';
                    },
                },
            },
            // optimize for production
            minify: 'terser',
            sourcemap: mode === 'development',
        },

        // allow environment variables starting with VITE_ to be exposed to the client
        envPrefix: ['VITE_'],

        // define the directory for the environment variables
        envDir: process.cwd(),

        plugins: [
            // compile React files faster using SWC
            react(),
        ],

        // configure module resolution
        resolve: {
            alias: {
                // allow using '@' in imports instead of relative paths to the src folder
                '@': path.resolve(__dirname, './src'),
            },
        },

        // development server configuration
        server: {
            port: 3000,
            open: true,
            // proxy API calls to your translation service
            proxy: {
                '/api': {
                    target: 'http://localhost:8080',
                    changeOrigin: true,
                    rewrite: (path) => path.replace(/^\/api/, ''),
                },
            },
        },

        // preview server configuration
        preview: {
            port: 3000,
            open: true,
        },
    };
});
