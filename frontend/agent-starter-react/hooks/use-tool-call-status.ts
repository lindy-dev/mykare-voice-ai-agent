'use client';

import { useEffect, useState } from 'react';
import { useRoomContext } from '@livekit/components-react';

export interface ToolStatus {
  tool: string;
  status: 'started' | 'completed' | 'error';
  message: string;
}

export function useToolCallStatus() {
  const room = useRoomContext();
  const [toolStatus, setToolStatus] = useState<ToolStatus | null>(null);

  useEffect(() => {
    if (!room) return;

    const handler = async (data: {
      requestId: string;
      callerIdentity: string;
      payload: string;
    }) => {
      const parsed: ToolStatus = JSON.parse(data.payload);
      setToolStatus(parsed);

      // Auto-clear after completion/error
      if (parsed.status === 'completed' || parsed.status === 'error') {
        setTimeout(() => setToolStatus(null), 2000);
      }

      return 'ok';
    };

    room.registerRpcMethod('tool_status', handler);
    return () => {
      room.unregisterRpcMethod('tool_status');
    };
  }, [room]);

  return { toolStatus };
}
