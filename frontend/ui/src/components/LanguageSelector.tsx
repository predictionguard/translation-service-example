import { useState } from 'react';
import {
  Box,
  Button,
  MenuItem,
  IconButton,
  Typography,
  TextField,
  InputAdornment,
  Menu,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import CheckIcon from '@mui/icons-material/Check';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';

interface Language {
  code: string;
  name: string;
}

interface LanguageSelectorProps {
  languages: Language[];
  selectedLanguage: string;
  onLanguageChange: (code: string) => void;
  quickLanguages: string[];
  placeholder?: string;
}

export default function LanguageSelector({
  languages,
  selectedLanguage,
  onLanguageChange,
  quickLanguages,
  placeholder = "Search languages"
}: LanguageSelectorProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const filteredLanguages = languages.filter(lang =>
    lang.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleOpenMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleCloseMenu = () => {
    setAnchorEl(null);
    setSearchTerm('');
  };

  const handleSelectLanguage = (code: string) => {
    onLanguageChange(code);
    handleCloseMenu();
  };

  const getLanguageName = (code: string) => {
    return languages.find(lang => lang.code === code)?.name || code;
  };

  return (
    <Box sx={{ flex: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
        {quickLanguages.map((code) => (
          <Button
            key={code}
            variant={selectedLanguage === code ? 'contained' : 'outlined'}
            onClick={() => onLanguageChange(code)}
            size="small"
          >
            {getLanguageName(code)}
          </Button>
        ))}
        <IconButton
          onClick={handleOpenMenu}
          size="small"
          sx={{
            border: 1,
            borderColor: 'divider',
            borderRadius: 1,
          }}
        >
          <KeyboardArrowDownIcon fontSize="small" />
        </IconButton>
      </Box>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleCloseMenu}
        PaperProps={{
          sx: {
            maxHeight: 500,
            width: 600,
            mt: 1,
          },
        }}
      >
        <Box sx={{ p: 2, pb: 1 }}>
          <TextField
            size="small"
            autoFocus
            placeholder={placeholder}
            fullWidth
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
            }}
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </Box>
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: 0.5,
            p: 2,
            pt: 1,
          }}
        >
          {filteredLanguages.map((lang) => (
            <MenuItem
              key={lang.code}
              onClick={() => handleSelectLanguage(lang.code)}
              sx={{
                borderRadius: 1,
                display: 'flex',
                justifyContent: 'space-between',
                '&:hover': {
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <Typography variant="body2">{lang.name}</Typography>
              {selectedLanguage === lang.code && (
                <CheckIcon fontSize="small" color="primary" />
              )}
            </MenuItem>
          ))}
        </Box>
      </Menu>
    </Box>
  );
}

