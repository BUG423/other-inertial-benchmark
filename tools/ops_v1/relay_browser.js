#!/usr/bin/env node
/*
 * Bounded, read-only browser client for app-remote-terminal.
 *
 * It creates one temporary PTY through the relay, runs the supplied command,
 * waits for a private completion marker, deletes the PTY, and emits one JSON
 * result.  Tokens are read from a 0600 pairing profile and are never printed.
 */

'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

function loadWebSocket() {
  const candidates = [
    process.env.REMOTE_TERMINAL_WS_MODULE,
    path.resolve(__dirname, '../../../app-remote-terminal/agent/node_modules/ws'),
    path.resolve(__dirname, '../../../app-remote-terminal/server/node_modules/ws'),
  ].filter(Boolean);
  let lastError;
  for (const candidate of candidates) {
    try { return require(candidate); } catch (error) { lastError = error; }
  }
  if (typeof WebSocket === 'function') return WebSocket;
  throw lastError || new Error('ws module is not available');
}

function argument(name, fallback = '') {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] || fallback : fallback;
}

function fail(message) {
  throw new Error(message);
}

function readProfile(profilePath) {
  const stat = fs.statSync(profilePath);
  if ((stat.mode & 0o077) !== 0) fail('pairing profile must be mode 0600');
  const profile = JSON.parse(fs.readFileSync(profilePath, 'utf8'));
  if (profile.type !== 'remote-terminal-app') fail('invalid app pairing profile type');
  const relayUrl = new URL(profile.relayUrl);
  if (relayUrl.protocol !== 'wss:') fail('relayUrl must use wss://');
  if (relayUrl.search || relayUrl.hash || !profile.browserToken ||
      typeof profile.browserToken !== 'string' || profile.browserToken.length < 32) {
    fail('invalid app pairing profile');
  }
  return { relayUrl: relayUrl.toString(), browserToken: profile.browserToken };
}

function send(ws, message) {
  ws.send(JSON.stringify(message));
}

async function main() {
  const profilePath = argument('--profile');
  const command = argument('--command');
  const timeoutMs = Math.max(5000, Math.min(180000, Number(argument('--timeout', '90')) * 1000));
  if (!profilePath || !command) fail('usage: relay_browser.js --profile PROFILE --command COMMAND [--timeout SECONDS]');
  const profile = readProfile(profilePath);
  const WebSocket = loadWebSocket();
  const sessionId = `relay-read-${Date.now().toString(36)}-${crypto.randomBytes(8).toString('hex')}`;
  const marker = `__REMOTE_RELAY_DONE_${crypto.randomBytes(12).toString('hex')}__:`;
  const encoded = Buffer.from(command, 'utf8').toString('base64');
  const remoteCommand = `printf '%s' ${encoded} | base64 -d | bash; rc=$?; printf '\\n${marker}%s\\n' "$rc"\r`;
  const startedAt = Date.now();

  return await new Promise((resolve, reject) => {
    let ws;
    let settled = false;
    let authenticated = false;
    let created = false;
    let attached = false;
    let deleting = false;
    let output = '';
    let timer;

    const finish = (error, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { if (ws && ws.readyState === WebSocket.OPEN) ws.close(); } catch {}
      if (error) reject(error); else resolve(value);
    };
    const cleanup = () => {
      if (!ws || ws.readyState !== WebSocket.OPEN || !created || deleting) return;
      deleting = true;
      try { send(ws, { type: 'terminal_delete', sessionId }); } catch {}
    };
    const onMessage = (raw) => {
      let message;
      try { message = JSON.parse(raw.toString()); } catch { return; }
      if (message.type === 'error' && !authenticated) {
        finish(new Error(`relay authentication failed: ${message.message || 'unknown error'}`));
        return;
      }
      if (message.type === 'auth_ok') {
        authenticated = true;
        if (message.agentOnline !== true) {
          finish(new Error('remote Agent is offline'));
          return;
        }
        send(ws, { type: 'terminal_create', sessionId, title: 'read-only status reporter' });
        return;
      }
      if (message.sessionId !== sessionId) return;
      if (message.type === 'terminal_error') {
        cleanup();
        finish(new Error(`remote terminal error: ${message.message || 'unknown error'}`));
        return;
      }
      if (message.type === 'terminal_created') {
        created = true;
        send(ws, { type: 'terminal_attach', sessionId });
        return;
      }
      if (message.type === 'terminal_attached') {
        attached = true;
        send(ws, { type: 'terminal_input', sessionId, data: remoteCommand });
        return;
      }
      if (message.type === 'terminal_snapshot' || message.type === 'terminal_output') {
        if (!attached && message.type === 'terminal_snapshot') return;
        output += String(message.data || '');
        if (Buffer.byteLength(output, 'utf8') > 8 * 1024 * 1024) {
          cleanup();
          finish(new Error('remote command output exceeds 8 MiB bound'));
          return;
        }
        let index = -1;
        let exitCode;
        let searchFrom = 0;
        while (true) {
          const candidate = output.indexOf(marker, searchFrom);
          if (candidate < 0) break;
          const tail = output.slice(candidate + marker.length);
          const match = tail.match(/^(?:\r?\n)?(-?\d+)/);
          if (match) {
            index = candidate;
            exitCode = Number(match[1]);
            break;
          }
          // The first occurrence is normally the PTY echo of the command;
          // keep looking for the marker printed by the remote shell.
          searchFrom = candidate + marker.length;
        }
        if (index < 0) return;
        cleanup();
        finish(null, {
          exitCode,
          output: output.slice(0, index),
          sessionId,
          elapsedMs: Date.now() - startedAt,
        });
      }
    };

    timer = setTimeout(() => {
      cleanup();
      finish(new Error('relay command timed out'));
    }, timeoutMs);
    try {
      ws = new WebSocket(profile.relayUrl);
      ws.on('open', () => send(ws, {
        type: 'auth', token: profile.browserToken, role: 'browser', protocolVersion: 2,
      }));
      ws.on('message', onMessage);
      ws.on('error', (error) => finish(new Error(`relay websocket error: ${error.message}`)));
      ws.on('close', (code, reason) => {
        if (!settled) finish(new Error(`relay websocket closed (${code})${reason ? `: ${reason}` : ''}`));
      });
    } catch (error) {
      finish(error);
    }
  });
}

main().then((result) => {
  process.stdout.write(`${JSON.stringify({ ok: true, ...result })}\n`);
}).catch((error) => {
  process.stderr.write(`${JSON.stringify({ ok: false, error: error.message })}\n`);
  process.exitCode = 1;
});
