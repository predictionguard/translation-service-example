import {useQuery, useMutation} from '@tanstack/react-query';

export interface TranslationResult {
    translations: Array<{
        translation: string;
        score: number;
        model: string;
        status: string;
    }>;
    best_translation: string;
    best_score: number;
    best_translation_model: string;
}

export interface TranslationRequest {
    text: string;
    source_lang: string;
    target_lang: string;
    model: string;
}

export interface Model {
    code: string;
    name: string;
}

// Hook to fetch available models
export const useModels = (apiUrl: string = '/api') => {
    return useQuery({
        queryKey: ['models'],
        queryFn: async (): Promise<Model[]> => {
            const response = await fetch(`${apiUrl}/models`);
            if (!response.ok) {
                throw new Error('Failed to fetch models');
            }
            const data = await response.json();
            return data.models.map((model: string) => ({
                code: model,
                name: model.split('__').pop()?.toUpperCase().replace(/-/g, ' ') || model.toUpperCase().replace(/-/g, ' '),
            }));
        },
    });
};

// Hook for translation mutation
export const useTranslation = (apiUrl: string = '/api') => {
    return useMutation({
        mutationFn: async (request: TranslationRequest): Promise<TranslationResult> => {
            console.log(request);
            const response = await fetch(`${apiUrl}/translate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(request),
            });

            if (!response.ok) {
                throw new Error(`Translation failed: ${response.statusText}`);
            }

            return response.json();
        },
    });
};
