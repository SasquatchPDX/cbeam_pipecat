import "@fontsource-variable/geist";
import "@fontsource-variable/geist-mono";

import { PipecatClientProvider } from "@pipecat-ai/client-react";
import { LanguageProvider } from "./contexts/LanguageContext";
import SimpleVoiceUI from "./SimpleVoiceUI";

export default function App() {
  // AudioClientHelper provides its own client, so we wrap the app with an empty provider
  return (
    <PipecatClientProvider>
      <LanguageProvider>
        <SimpleVoiceUI />
      </LanguageProvider>
    </PipecatClientProvider>
  );
}