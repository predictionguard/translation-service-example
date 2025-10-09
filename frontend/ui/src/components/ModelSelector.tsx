import { useState } from 'react';
import {
  Box,
  Button,
  MenuItem,
  Typography,
  TextField,
  InputAdornment,
  Menu,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import CheckIcon from '@mui/icons-material/Check';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';

interface Model {
  code: string;
  name: string;
}

interface ModelSelectorProps {
  models: Model[];
  selectedModel: string;
  onModelChange: (code: string) => void;
}

export default function ModelSelector({
  models,
  selectedModel,
  onModelChange,
}: ModelSelectorProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const filteredModels = models.filter(model =>
    model.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleOpenMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleCloseMenu = () => {
    setAnchorEl(null);
    setSearchTerm('');
  };

  const handleSelectModel = (code: string) => {
    onModelChange(code);
    handleCloseMenu();
  };

  const getModelName = (code: string) => {
    return models.find(model => model.code === code)?.name || code;
  };

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Typography variant="body2" sx={{ color: 'text.secondary', fontWeight: 500 }}>
        Model:
      </Typography>
      <Button
        variant="outlined"
        onClick={handleOpenMenu}
        endIcon={<KeyboardArrowDownIcon />}
        size="small"
        sx={{ minWidth: 140 }}
      >
        {getModelName(selectedModel)}
      </Button>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleCloseMenu}
        PaperProps={{
          sx: {
            maxHeight: 400,
            width: 300,
            mt: 1,
          },
        }}
      >
        <Box sx={{ p: 2, pb: 1 }}>
          <TextField
            size="small"
            autoFocus
            placeholder="Search models"
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
        <Box sx={{ p: 2, pt: 1 }}>
          {filteredModels.map((model) => (
            <MenuItem
              key={model.code}
              onClick={() => handleSelectModel(model.code)}
              sx={{
                borderRadius: 1,
                display: 'flex',
                justifyContent: 'space-between',
                mb: 0.5,
                '&:hover': {
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <Typography variant="body2">{model.name}</Typography>
              {selectedModel === model.code && (
                <CheckIcon fontSize="small" color="primary" />
              )}
            </MenuItem>
          ))}
        </Box>
      </Menu>
    </Box>
  );
}

