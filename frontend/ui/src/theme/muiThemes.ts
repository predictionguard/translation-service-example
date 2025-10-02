import {createTheme} from '@mui/material/styles';

export const createSystemTheme = (mode: 'light' | 'dark') => {
    return createTheme({
        palette: {
            mode,
            primary: {
                main: mode === 'light' ? '#3B82F6' : '#60A5FA',
                light: mode === 'light' ? '#60A5FA' : '#93C5FD',
                dark: mode === 'light' ? '#2563EB' : '#3B82F6',
            },
            secondary: {
                main: mode === 'light' ? '#64748B' : '#94A3B8',
            },
            background: {
                default: mode === 'light' ? '#F8FAFC' : '#0F172A',
                paper: mode === 'light' ? '#FFFFFF' : '#1E293B',
            },
            text: {
                primary: mode === 'light' ? '#0F172A' : '#F1F5F9',
                secondary: mode === 'light' ? '#64748B' : '#94A3B8',
            },
        },
        shape: {
            borderRadius: 12,
        },
        typography: {
            fontFamily: '"Inter", system-ui, -apple-system, sans-serif',
            h4: {
                fontWeight: 600,
                letterSpacing: '-0.025em',
            },
            h6: {
                fontWeight: 600,
            },
        },
        components: {
            MuiPaper: {
                styleOverrides: {
                    root: {
                        backgroundImage: 'none',
                        borderRadius: 12,
                        border: mode === 'light' ? '1px solid #E2E8F0' : '1px solid #334155',
                        boxShadow: mode === 'light' ? '0 1px 3px rgba(0, 0, 0, 0.05)' : '0 4px 6px -1px rgba(0, 0, 0, 0.3)',
                    },
                },
            },
            MuiButton: {
                styleOverrides: {
                    root: {
                        textTransform: 'none',
                        fontWeight: 600,
                        borderRadius: 8,
                    },
                },
            },
            MuiTextField: {
                styleOverrides: {
                    root: {
                        '& .MuiOutlinedInput-root': {
                            backgroundColor: mode === 'light' ? '#FFFFFF' : '#0F172A',
                            '& fieldset': {
                                borderColor: mode === 'light' ? '#E2E8F0' : '#334155',
                            },
                        },
                    },
                },
            },
        },
    });
};
