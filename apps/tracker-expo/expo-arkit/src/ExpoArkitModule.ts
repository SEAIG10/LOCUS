import { NativeModule, requireNativeModule } from 'expo';

import { ExpoArkitModuleEvents } from './ExpoArkit.types';

declare class ExpoArkitModule extends NativeModule<ExpoArkitModuleEvents> {
  PI: number;
  hello(): string;
  setValueAsync(value: string): Promise<void>;
}

// This call loads the native module object from the JSI.
export default requireNativeModule<ExpoArkitModule>('ExpoArkit');
