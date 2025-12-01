import { registerWebModule, NativeModule } from 'expo';

import { ExpoArkitModuleEvents } from './ExpoArkit.types';

class ExpoArkitModule extends NativeModule<ExpoArkitModuleEvents> {
  PI = Math.PI;
  async setValueAsync(value: string): Promise<void> {
    this.emit('onChange', { value });
  }
  hello() {
    return 'Hello world! 👋';
  }
}

export default registerWebModule(ExpoArkitModule, 'ExpoArkitModule');
