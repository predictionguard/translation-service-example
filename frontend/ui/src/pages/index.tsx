import { useMemo } from 'react';
import { ThemeProvider, CssBaseline, useMediaQuery } from '@mui/material';
import { Container, Box, Typography } from '@mui/material';
import TranslationInterface from './TranslationInterface';
import { createSystemTheme } from '../theme/muiThemes';

const Index = () => {
  const prefersDarkMode = useMediaQuery('(prefers-color-scheme: dark)');

  const theme = useMemo(() => {
    return createSystemTheme(prefersDarkMode ? 'dark' : 'light');
  }, [prefersDarkMode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box
        sx={{
          minHeight: '100vh',
          background: theme.palette.background.default,
        }}
      >
        <Box sx={{ py: 3, px: 4 }}>
          <Typography 
            variant="h6" 
            component="div" 
            sx={{ 
              fontWeight: 500,
              fontSize: '1.125rem',
              color: 'text.primary',
              letterSpacing: '-0.01em'
            }}
          >
            Translation Service
          </Typography>
        </Box>

        <Container maxWidth="xl" sx={{ py: 6 }}>
          <Box sx={{ mb: 6, textAlign: 'center' }}>
            <Typography variant="h4" gutterBottom>
              Multi-Engine Translation
            </Typography>
          </Box>

          <TranslationInterface />
        </Container>
      </Box>
    </ThemeProvider>
  );
};

export default Index;
