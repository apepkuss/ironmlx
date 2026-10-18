// Upstream browser client. IronMLX itself remains a headless inference service.
// Serve this module over HTTPS or localhost; import it from your application.
export async function connectTranscription({
  baseUrl = 'http://127.0.0.1:9068',
  apiKey,
  model = 'mlx-community/Qwen3-ASR-0.6B-8bit',
  language,
  diarization = null,
  turnDetection = { type: 'server_vad' },
  iceServers = [],
  iceTransportPolicy = 'all',
  stream,
  onEvent = () => {},
}) {
  const endpoint = new URL('/v1/realtime/calls', baseUrl);
  const headers = apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
  const ownedStream = !stream;
  const source = stream ?? await navigator.mediaDevices.getUserMedia({ audio: true });
  const tracks = source.getAudioTracks();
  let pc, channel;
  try {
    if (tracks.length !== 1) throw new Error('Exactly one audio track is required');
    pc = new RTCPeerConnection({ iceServers, iceTransportPolicy });
    channel = pc.createDataChannel('oai-events', { ordered: true });
  } catch (error) {
    pc?.close();
    if (ownedStream) source.getTracks().forEach(track => track.stop());
    throw error;
  }
  let callUrl;
  let closed = false;
  const dispose = () => {
    pc.close();
    if (ownedStream) source.getTracks().forEach(track => track.stop());
  };
  const hangup = async () => {
    if (closed) return;
    closed = true;
    try {
      if (callUrl) {
        const response = await fetch(callUrl, { method: 'POST', headers });
        if (!response.ok && response.status !== 404) throw new Error(`Hangup failed: ${response.status}`);
      }
    } finally { dispose(); }
  };
  channel.addEventListener('message', message => {
    try { onEvent(JSON.parse(message.data)); }
    catch (error) { console.error('Invalid Realtime event or client callback failure', error); }
  });
  const created = new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Realtime session creation timed out')), 90000);
    const listener = message => {
      try {
        const event = JSON.parse(message.data);
        if (event.type === 'session.created' || event.type === 'error') {
          clearTimeout(timer);
          channel.removeEventListener('message', listener);
          if (event.type === 'error') reject(new Error(event.error?.message ?? 'Session failed'));
          else resolve(event.session);
        }
      } catch (error) { clearTimeout(timer); reject(error); }
    };
    channel.addEventListener('message', listener);
    channel.addEventListener('close', () => { clearTimeout(timer); reject(new Error('Realtime data channel closed')); }, { once: true });
  });
  // Avoid an unhandled rejection if HTTP negotiation fails before awaiting this promise.
  created.catch(() => {});
  try {
    pc.addTrack(tracks[0], source);
    await pc.setLocalDescription(await pc.createOffer());
    if (pc.iceGatheringState !== 'complete') {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('Local ICE gathering timed out')), 15000);
        const listener = () => {
          if (pc.iceGatheringState === 'complete') {
            clearTimeout(timer); pc.removeEventListener('icegatheringstatechange', listener); resolve();
          }
        };
        pc.addEventListener('icegatheringstatechange', listener);
        listener();
      });
    }
    const transcription = { model };
    if (language) transcription.language = language;
    const input = { transcription, turn_detection: turnDetection };
    if (diarization !== null && diarization !== false) input.diarization = diarization === true ? {} : diarization;
    const form = new FormData();
    form.set('sdp', pc.localDescription.sdp);
    form.set('session', JSON.stringify({ type: 'transcription', audio: { input } }));
    const response = await fetch(endpoint, { method: 'POST', headers, body: form });
    if (!response.ok) throw new Error(await response.text());
    const location = response.headers.get('Location');
    if (!location) throw new Error('The call response is missing Location');
    const resource = new URL(location, endpoint);
    if (resource.origin !== endpoint.origin) throw new Error('Unexpected call resource origin');
    callUrl = new URL(`${resource.pathname}/hangup`, endpoint);
    await pc.setRemoteDescription({ type: 'answer', sdp: await response.text() });
    const session = await created;
    const send = event => {
      if (channel.readyState !== 'open') throw new Error('Realtime data channel is not open');
      channel.send(JSON.stringify(event));
    };
    return {
      session, peer: pc, channel, callId: resource.pathname.split('/').at(-1),
      update: settings => send({ type: 'session.update', session: settings }),
      commit: () => send({ type: 'input_audio_buffer.commit' }),
      clear: () => send({ type: 'input_audio_buffer.clear' }),
      hangup,
    };
  } catch (error) {
    await hangup().catch(() => {});
    throw error;
  }
}
