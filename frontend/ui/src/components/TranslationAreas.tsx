import { useState, memo } from 'react';
import {
  Box,
  TextField,
  IconButton,
  Chip,
  Snackbar,
  CircularProgress,
} from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

interface TranslationAreasProps {
  sourceText: string;
  translatedText: string;
  qualityScore: number | null;
  modelUsed: string;
  isLoading: boolean;
  onSourceTextChange: (text: string) => void;
}

function TranslationAreas({
  sourceText,
  translatedText,
  qualityScore,
  modelUsed,
  isLoading,
  onSourceTextChange,
}: TranslationAreasProps) {
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string }>({
    open: false,
    message: '',
  });

  const handleCopy = () => {
    navigator.clipboard.writeText(translatedText);
    setSnackbar({ open: true, message: 'Translation copied to clipboard' });
  };

  const getScoreColor = (score: number): "success" | "warning" | "error" | "default" => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <>
      <Box sx={{ display: 'flex', gap: 3, flexWrap: { xs: 'wrap', md: 'nowrap' } }}>
        {/* Source Area */}
        <Box sx={{ flex: 1, minWidth: { xs: '100%', md: 0 } }}>
          <TextField
            multiline
            rows={12}
            fullWidth
            variant="outlined"
            placeholder="Type or paste text here..."
            value={sourceText}
            onChange={(e) => onSourceTextChange(e.target.value)}
          />
        </Box>

        {/* Target Area */}
        <Box sx={{ flex: 1, minWidth: { xs: '100%', md: 0 }, position: 'relative' }}>
          <TextField
            multiline
            rows={12}
            fullWidth
            variant="outlined"
            placeholder={isLoading ? "" : "Translation will appear here..."}
            value={translatedText}
            InputProps={{
              readOnly: true,
              startAdornment: isLoading && (
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 1.5, 
                  color: 'text.secondary',
                  fontSize: '16px',
                  fontWeight: 400,
                  fontFamily: 'inherit'
                }}>
                  <CircularProgress size={20} sx={{ color: 'primary.main' }} />
                  <span>Translating...</span>
                </Box>
              ),
              endAdornment: translatedText && !isLoading && (
                <IconButton
                  size="small"
                  onClick={handleCopy}
                  sx={{ 
                    position: 'absolute',
                    bottom: 8,
                    right: 8,
                    zIndex: 1,
                    backgroundColor: 'background.paper',
                    '&:hover': {
                      backgroundColor: 'action.hover'
                    }
                  }}
                >
                  <ContentCopyIcon fontSize="small" />
                </IconButton>
              ),
            }}
          />
          {qualityScore !== null && (
            <Box sx={{ mt: 2, display: 'flex', gap: 1.5, alignItems: 'center' }}>
              <Chip
                label={`Quality: ${(qualityScore * 100).toFixed(0)}%`}
                color={getScoreColor(qualityScore)}
                size="small"
                sx={{
                  borderRadius: 2,
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  height: 28,
                  '& .MuiChip-label': {
                    px: 1.5,
                  },
                }}
              />
              {modelUsed && (
                <Chip
                  label={`Best: ${modelUsed}`}
                  color="default"
                  size="small"
                  variant="outlined"
                  sx={{
                    borderRadius: 2,
                    fontWeight: 500,
                    fontSize: '0.75rem',
                    height: 28,
                    borderColor: 'primary.main',
                    color: 'primary.main',
                    backgroundColor: 'transparent',
                    '& .MuiChip-label': {
                      px: 1.5,
                    },
                    '&:hover': {
                      backgroundColor: 'primary.50',
                      borderColor: 'primary.dark',
                    },
                  }}
                />
              )}
            </Box>
          )}
        </Box>
      </Box>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        message={snackbar.message}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      />
    </>
  );
}

export default memo(TranslationAreas);

