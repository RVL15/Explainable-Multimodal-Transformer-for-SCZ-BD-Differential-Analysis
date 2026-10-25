import React, { useRef, useEffect, useState } from 'react';
import { ChannelWaveform } from '../types';

interface WaveformCanvasProps {
  channels: ChannelWaveform[];
  timeVector: number[];
  selectedChannels?: string[];
  showProcessed?: boolean;
  amplitudeScale?: number; // scale multiplier
  height?: number;
}

export const WaveformCanvas: React.FC<WaveformCanvasProps> = ({
  channels,
  timeVector,
  selectedChannels,
  showProcessed = true,
  amplitudeScale = 1.0,
  height = 500,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{ time: number; channel: string; val: number } | null>(null);

  const activeChannels = selectedChannels && selectedChannels.length > 0
    ? channels.filter((c) => selectedChannels.includes(c.channel))
    : channels;

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || activeChannels.length === 0 || timeVector.length === 0) return;

    const width = container.clientWidth;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    ctx.clearRect(0, 0, width, height);

    const numCh = activeChannels.length;
    const channelHeight = height / (numCh + 1);
    const numPoints = timeVector.length;
    const xStep = (width - 70) / (numPoints - 1);

    // Draw background grid lines and channel base lines
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;

    for (let i = 0; i < numCh; i++) {
      const centerY = (i + 1) * channelHeight;
      ctx.beginPath();
      ctx.moveTo(60, centerY);
      ctx.lineTo(width - 10, centerY);
      ctx.stroke();

      // Channel label
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.fillStyle = '#475569';
      ctx.textAlign = 'right';
      ctx.fillText(activeChannels[i].channel, 52, centerY + 4);
    }

    // Time ticks at bottom
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    const numTicks = 6;
    for (let t = 0; t <= numTicks; t++) {
      const frac = t / numTicks;
      const x = 60 + frac * (width - 70);
      const timeVal = (timeVector[0] + frac * (timeVector[timeVector.length - 1] - timeVector[0])).toFixed(1);
      ctx.fillText(`${timeVal}s`, x, height - 8);
    }

    // Scale factor: microvolts to pixels
    const pxPerUv = (channelHeight * 0.45 * amplitudeScale) / 50;

    // Plot traces
    activeChannels.forEach((ch, idx) => {
      const centerY = (idx + 1) * channelHeight;

      // 1. Raw signal (slate color)
      if (ch.raw_samples && ch.raw_samples.length > 0) {
        ctx.beginPath();
        ctx.strokeStyle = ch.processed_samples && showProcessed ? '#cbd5e1' : '#3b82f6';
        ctx.lineWidth = ch.processed_samples && showProcessed ? 1 : 1.4;

        for (let p = 0; p < numPoints; p++) {
          const x = 60 + p * xStep;
          const y = centerY - (ch.raw_samples[p] || 0) * pxPerUv;
          if (p === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      // 2. Processed signal (teal/blue accent)
      if (showProcessed && ch.processed_samples && ch.processed_samples.length > 0) {
        ctx.beginPath();
        ctx.strokeStyle = '#0284c7';
        ctx.lineWidth = 1.6;

        for (let p = 0; p < numPoints; p++) {
          const x = 60 + p * xStep;
          // Scale processed z-score
          const y = centerY - (ch.processed_samples[p] || 0) * (channelHeight * 0.22 * amplitudeScale);
          if (p === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    });

    // 50 uV calibration scale bar in top right
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const scaleBarX = width - 40;
    const scaleBarY = 25;
    const scaleBarHeight = 50 * pxPerUv;
    ctx.moveTo(scaleBarX, scaleBarY - scaleBarHeight / 2);
    ctx.lineTo(scaleBarX, scaleBarY + scaleBarHeight / 2);
    ctx.stroke();
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillStyle = '#0f172a';
    ctx.textAlign = 'right';
    ctx.fillText('50 µV', scaleBarX - 6, scaleBarY + 3);

  }, [activeChannels, timeVector, showProcessed, amplitudeScale, height]);

  return (
    <div ref={containerRef} className="w-full bg-white rounded-xl border border-slate-200 p-3 relative overflow-hidden shadow-xs">
      <div className="flex items-center justify-between text-xs text-slate-500 mb-2 px-2">
        <div className="flex items-center space-x-4">
          <span className="flex items-center">
            <span className="w-3 h-0.5 bg-slate-300 mr-1.5 inline-block" />
            Raw Signal
          </span>
          {showProcessed && (
            <span className="flex items-center">
              <span className="w-3 h-1 bg-sky-600 mr-1.5 inline-block rounded-full" />
              1-45 Hz Bandpassed + CAR
            </span>
          )}
        </div>
        <span className="font-mono text-slate-400">Duration: {timeVector.length > 0 ? `${(timeVector[timeVector.length - 1] - timeVector[0]).toFixed(1)}s` : '0s'}</span>
      </div>

      <canvas ref={canvasRef} className="cursor-crosshair w-full block" />
    </div>
  );
};
