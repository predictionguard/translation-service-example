# Multi-Engine Translation Service

A modern, user-friendly translation interface built with React, TypeScript, Vite, and Material-UI. This application provides seamless translation between multiple languages using various AI models.

## Features

- 🌍 **Multi-language Support**: Translate between 80+ languages
- 🤖 **Multiple AI Models**: Choose from OPUS-MT, mBART-50, M2M100, NLLB, and Helsinki-NLP
- ⚡ **Real-time Translation**: Auto-translate as you type with debouncing
- 📱 **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- 🎨 **Modern UI**: Clean, intuitive interface built with Material-UI
- 📊 **Quality Scores**: See translation confidence scores
- 🔄 **Language Swap**: Quick swap between source and target languages
- 📋 **Copy to Clipboard**: Easy copying of translated text

## Quick Start

### Prerequisites

- Node.js (version 16 or higher)
- npm or yarn
- Translation service backend running on port 8080

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm run dev
   ```

3. **Open your browser** and navigate to `http://localhost:3000`

### Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Usage

1. **Select Languages**: Choose your source and target languages using the quick-select buttons or dropdown menus
2. **Choose Model**: Select your preferred translation model from the dropdown
3. **Type Text**: Enter text in the source language field
4. **View Translation**: The translation will appear automatically in the target field
5. **Copy Result**: Click the copy button to copy the translation to your clipboard

## API Integration

The frontend expects a translation service backend running on `http://localhost:8080` with the following endpoints:

- `GET /models` - Returns available translation models
- `POST /translate` - Translates text with the following payload:
  ```json
  {
    "text": "Hello world",
    "source_lang": "eng",
    "target_lang": "fra",
    "model": "opus-mt"
  }
  ```

## Configuration

### Environment Variables

- `VITE_API_URL`: Override the default API URL (defaults to `/api`)

### Customization

- **Languages**: Edit `src/data/languages.ts` to add or modify supported languages
- **Models**: Edit `src/data/models.ts` to add or modify available models
- **Styling**: Modify the theme in `src/App.tsx` to customize colors and typography

## Technology Stack

- **React 19** - Modern React with hooks
- **TypeScript** - Type-safe JavaScript
- **Vite** - Fast build tool and dev server
- **Material-UI** - React component library
- **Emotion** - CSS-in-JS styling

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Project Structure

```
src/
├── components/
│   └── TranslationInterface.tsx  # Main translation component
├── data/
│   ├── languages.ts             # Supported languages
│   └── models.ts                # Available models
├── App.tsx                      # Main app component
└── main.tsx                     # App entry point
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License.
