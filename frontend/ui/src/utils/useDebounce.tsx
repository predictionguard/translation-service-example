import {useRef, useEffect, useMemo} from 'react';

// Simple debounce utility function
function debounce<T extends (...args: unknown[]) => unknown>(func: T, delay: number): (...args: Parameters<T>) => void {
    let timeoutId: NodeJS.Timeout;

    return (...args: Parameters<T>) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func(...args), delay);
    };
}

export function useDebounce<T extends (...args: unknown[]) => unknown>(callback: T, delay: number, deps: React.DependencyList = []): (...args: Parameters<T>) => void {
    // Create ref to hold the latest callback
    const callbackRef = useRef(callback);

    // Update ref when dependencies change
    useEffect(() => {
        callbackRef.current = callback;
    }, [deps, callback]);

    // Create debounced function only once
    const debouncedCallback = useMemo(() => {
        const func = (...args: Parameters<T>) => {
            callbackRef.current(...args);
        };
        return debounce(func, delay);
    }, [delay]);

    return debouncedCallback;
}
