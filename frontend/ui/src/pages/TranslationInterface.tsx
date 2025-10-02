import { useState, useEffect, useCallback } from 'react';
import { Box, Alert, Tooltip, IconButton } from '@mui/material';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import { languages } from '../components/languages';
import { useModels, useTranslation } from '../hooks/useTranslation';
import { useDebounce } from '../utils/useDebounce';
import LanguageSelector from '../components/LanguageSelector';
import ModelSelector from '../components/ModelSelector';
import TranslationAreas from '../components/TranslationAreas';

interface TranslationInterfaceProps {
  apiUrl?: string;
}

export default function TranslationInterface({ apiUrl = '/api' }: TranslationInterfaceProps) {
  const [sourceText, setSourceText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [sourceLang, setSourceLang] = useState('eng');
  const [targetLang, setTargetLang] = useState('fra');
  const [selectedModel, setSelectedModel] = useState('');
  const [qualityScore, setQualityScore] = useState<number | null>(null);
  const [modelUsed, setModelUsed] = useState<string>('');
  const [error, setError] = useState<string>('');

  // Quick selection languages 
  const [quickLanguages, setQuickLanguages] = useState(['eng', 'spa', 'fra']);

  // Use React Query hooks
  const { data: availableModels = [] } = useModels(apiUrl);
  const translationMutation = useTranslation(apiUrl);
  const isLoading = translationMutation.isPending;

  // Set default model when models are loaded
  useEffect(() => {
    if (availableModels.length > 0 && !selectedModel) {
      setSelectedModel(availableModels[0].code);
    }
  }, [availableModels, selectedModel]);

  const handleTranslate = useCallback(async () => {
    if (!sourceText.trim()) {
      return;
    }

    setError('');
    setTranslatedText('');
    setQualityScore(null);

    try {
      const data = await translationMutation.mutateAsync({
        text: sourceText,
        source_lang: sourceLang,
        target_lang: targetLang,
        model: selectedModel,
      });
      
      setTranslatedText(data.best_translation);
      setQualityScore(data.best_score);
      setModelUsed(data.best_translation_model);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Translation failed';
      setError(errorMessage);
    }
  }, [sourceText, sourceLang, targetLang, selectedModel, translationMutation]);

  // Create debounced translation function
  const debouncedTranslate = useDebounce(handleTranslate, 800, [sourceText, sourceLang, targetLang, selectedModel]);

  // Auto-translate with debouncing
  useEffect(() => {
    if (!sourceText.trim()) {
      setTranslatedText('');
      setQualityScore(null);
      setModelUsed('');
      return;
    }

    debouncedTranslate();
  }, [sourceText, sourceLang, targetLang, selectedModel, debouncedTranslate]);

  const handleSwapLanguages = () => {
    const temp = sourceLang;
    setSourceLang(targetLang);
    setTargetLang(temp);
    setSourceText(translatedText);
    setTranslatedText('');
    setQualityScore(null);
  };

  const handleLanguageChange = (languageCode: string, isSource: boolean) => {
    if (isSource) {
      setSourceLang(languageCode);
    } else {
      setTargetLang(languageCode);
    }
    
    // Update quick languages: add the new language to the front and remove the one at the end
    setQuickLanguages(prev => {
      const newQuickLanguages = [...prev];
      const existingIndex = newQuickLanguages.indexOf(languageCode);
      
      if (existingIndex === -1) {
        // Language not in quick list, add it to the front and remove the last one
        newQuickLanguages.pop();
        newQuickLanguages.unshift(languageCode);
      } else {
        // Language already in quick list, move it to the front
        newQuickLanguages.splice(existingIndex, 1);
        newQuickLanguages.unshift(languageCode);
      }
      
      return newQuickLanguages;
    });
  };

  return (
    <Box sx={{ width: '100%', maxWidth: 1400, mx: 'auto' }}>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Language Selectors and Model Selector */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 3, mb: 3, flexWrap: 'wrap' }}>
        <ModelSelector
          models={availableModels}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
        />
        
        <LanguageSelector
          languages={languages}
          selectedLanguage={sourceLang}
          onLanguageChange={(code) => handleLanguageChange(code, true)}
          quickLanguages={quickLanguages}
        />

        {/* Swap Button */}
        <Tooltip title="Swap Languages">
          <IconButton
            onClick={handleSwapLanguages}
            color="primary"
            sx={{
              flexShrink: 0,
              border: 1,
              borderColor: 'divider',
            }}
          >
            <SwapHorizIcon />
          </IconButton>
        </Tooltip>

        <LanguageSelector
          languages={languages}
          selectedLanguage={targetLang}
          onLanguageChange={(code) => handleLanguageChange(code, false)}
          quickLanguages={quickLanguages}
        />
      </Box>

      <TranslationAreas
        sourceText={sourceText}
        translatedText={translatedText}
        qualityScore={qualityScore}
        modelUsed={modelUsed}
        isLoading={isLoading}
        onSourceTextChange={setSourceText}
      />
    </Box>
  );
}
