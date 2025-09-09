import pyttsx3

class SimpleTTSService:
    def __init__(self, voice=None, rate=200, sample_rate=24000):
        self.engine = pyttsx3.init()
        if voice:
            self.engine.setProperty('voice', voice)
        self.engine.setProperty('rate', rate)
        self.sample_rate = sample_rate

    def __call__(self, text, *args, **kwargs):
        self.engine.say(text)
        self.engine.runAndWait()
        return text  # For compatibility

    def link(self, next_processor):
        # No-op for compatibility with pipeline
        pass

    async def setup(self, setup):
        # No-op async setup for pipeline compatibility
        pass

    async def queue_frame(self, frame, direction):
        # Only speak if not a system message
        role = getattr(frame, 'role', None)
        if role == 'system':
            return
        text = getattr(frame, 'text', str(frame))
        self.__call__(text)
        # No downstream processor, so nothing to return

    async def cleanup(self):
        # No-op async cleanup for pipeline compatibility
        pass
