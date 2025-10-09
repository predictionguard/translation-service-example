import React from 'react';

export function useEffectOnce(effect: React.EffectCallback, condition = true): void {
  const hasRun = React.useRef<boolean>(false);
  React.useEffect(() => {
    if (!hasRun.current && condition) {
      effect();
      hasRun.current = true;
    }
  }, [condition, effect, hasRun]);
}
