import * as React from 'react';

import { ExpoArkitViewProps } from './ExpoArkit.types';

export default function ExpoArkitView(props: ExpoArkitViewProps) {
  return (
    <div>
      <iframe
        style={{ flex: 1 }}
        src={props.url}
        onLoad={() => props.onLoad({ nativeEvent: { url: props.url } })}
      />
    </div>
  );
}
