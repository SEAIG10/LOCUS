import { requireNativeView } from 'expo';
import * as React from 'react';

import { ExpoArkitViewProps } from './ExpoArkit.types';

const NativeView: React.ComponentType<ExpoArkitViewProps> =
  requireNativeView('ExpoArkit');

export default function ExpoArkitView(props: ExpoArkitViewProps) {
  return <NativeView {...props} />;
}
