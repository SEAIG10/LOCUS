import { useCallback, useEffect, useRef, useState } from 'react';
import { TrackingStatus, ExtendedPosition } from '../types';

interface UseGeolocationOptions {
  onPosition?: (position: ExtendedPosition) => void;
  onError?: (error: GeolocationPositionError) => void;
  enableHighAccuracy?: boolean;
  timeout?: number;
  maximumAge?: number;
}

export const useGeolocation = (options: UseGeolocationOptions = {}) => {
  const [status, setStatus] = useState<TrackingStatus>('idle');
  const [position, setPosition] = useState<ExtendedPosition | null>(null);
  const [error, setError] = useState<GeolocationPositionError | null>(null);
  const watchIdRef = useRef<number | null>(null);

  const {
    onPosition,
    onError,
    enableHighAccuracy = true,
    timeout = 5000,
    maximumAge = 0,
  } = options;

  const handleSuccess = useCallback(
    (pos: GeolocationPosition) => {
      const extendedPos = pos as ExtendedPosition;
      setPosition(extendedPos);
      setError(null);
      onPosition?.(extendedPos);
    },
    [onPosition]
  );

  const handleError = useCallback(
    (err: GeolocationPositionError) => {
      setError(err);
      setStatus('error');
      onError?.(err);
    },
    [onError]
  );

  const startTracking = useCallback(() => {
    if (!navigator.geolocation) {
      const err = {
        code: 0,
        message: 'Geolocation is not supported by this browser',
        PERMISSION_DENIED: 1,
        POSITION_UNAVAILABLE: 2,
        TIMEOUT: 3,
      } as GeolocationPositionError;
      handleError(err);
      return;
    }

    if (watchIdRef.current !== null) {
      console.warn('Geolocation tracking is already active');
      return;
    }

    setStatus('tracking');

    watchIdRef.current = navigator.geolocation.watchPosition(
      handleSuccess,
      handleError,
      {
        enableHighAccuracy,
        timeout,
        maximumAge,
      }
    );
  }, [enableHighAccuracy, timeout, maximumAge, handleSuccess, handleError]);

  const stopTracking = useCallback(() => {
    if (watchIdRef.current !== null) {
      navigator.geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
    }
    setStatus('idle');
  }, []);

  useEffect(() => {
    return () => {
      if (watchIdRef.current !== null) {
        navigator.geolocation.clearWatch(watchIdRef.current);
      }
    };
  }, []);

  return {
    status,
    position,
    error,
    startTracking,
    stopTracking,
    isTracking: status === 'tracking',
  };
};
