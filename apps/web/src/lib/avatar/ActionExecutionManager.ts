/**
 * ActionExecutionManager.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Event-Driven Action Execution & Lifecycle Manager for SignBridge2Vision.
 *
 * Implements strict lifecycle state machine:
 *   IDLE → DETECTED → VALIDATED → QUEUED → EXECUTING → COMPLETED → IDLE
 *
 * Guarantees:
 * 1. Strict Idempotency: Every unique eventId executes at most once.
 * 2. Infinite Loop Elimination: Continuous static hand poses in camera frame do
 *    NOT cause repeated animation restarts or stuttering.
 * 3. Intentional Repetition: After an action completes and hands return to neutral
 *    or pause >= 900ms, repeated gestures trigger cleanly and execute once.
 * 4. Stale Event Cancellation: Old queued actions (>5000ms) are safely discarded.
 * 5. T-Pose Prevention: Always seamlessly returns to the natural IDLE breathing loop.
 */

import { ActionEvent, ActionLifecycleState } from '@signbridge/types';

export type ActionEventListener = (event: ActionEvent, previousState: ActionLifecycleState) => void;

export interface ActionDispatcher {
  playClip(
    actionId: string,
    options: { loop: boolean; crossFadeDuration: number },
    onFinished: () => void
  ): boolean;
  returnToIdle(crossFadeDuration?: number): void;
}

export class ActionExecutionManager {
  private currentState: ActionLifecycleState = 'IDLE';
  private activeEvent: ActionEvent | null = null;
  private queue: ActionEvent[] = [];
  private processedEventIds = new Set<string>();
  private lastExecutedSign: { sign: string; completedAt: number; eventId: string } | null = null;
  private listeners: Set<ActionEventListener> = new Set();
  private dispatcher: ActionDispatcher | null = null;
  private executionTimeoutTimer: any = null;
  public onSequenceComplete?: (sequenceId?: string) => void;
  private activeSequenceId: string | null = null;

  private readonly MAX_QUEUE_AGE_MS = 5000;
  private readonly REPETITION_COOLDOWN_MS = 900;

  constructor(dispatcher?: ActionDispatcher) {
    if (dispatcher) {
      this.dispatcher = dispatcher;
    }
  }

  public setDispatcher(dispatcher: ActionDispatcher) {
    this.dispatcher = dispatcher;
  }

  public subscribe(listener: ActionEventListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  public getState(): ActionLifecycleState {
    return this.currentState;
  }

  public getActiveEvent(): ActionEvent | null {
    return this.activeEvent;
  }

  public getQueueLength(): number {
    return this.queue.length;
  }

  /**
   * Primary entry point for recognized signs or speech commands.
   * Enforces DETECTED -> VALIDATED -> QUEUED lifecycle.
   */
  public submitEvent(
    actionId: string,
    options: {
      eventId?: string;
      confidence?: number;
      source?: 'vision' | 'speech' | 'ui' | 'system';
      metadata?: Record<string, any>;
      force?: boolean;
    } = {}
  ): ActionEvent | null {
    const canonical = actionId.toUpperCase().trim().replace(/\s+/g, '_');
    if (!canonical || canonical === 'IDLE' || canonical === 'SBV_IDLE' || canonical === 'NONE' || canonical === 'UNKNOWN') {
      return null;
    }

    const eventId = options.eventId || `evt_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const now = Date.now();

    // 1. Lifecycle: DETECTED
    const event: ActionEvent = {
      eventId,
      actionId: canonical,
      displayName: canonical.replace(/_/g, ' '),
      timestamp: now,
      confidence: options.confidence ?? 0.95,
      source: options.source ?? 'vision',
      state: 'DETECTED',
      metadata: options.metadata,
    };

    // 2. Validation Checks (Duplicate Prevention & Loop Elimination)
    if (this.processedEventIds.has(eventId)) {
      // Event already processed or in-flight (Idempotency guarantee)
      return null;
    }

    // Check if same sign is actively EXECUTING right now (allow if force=true for sequential queues)
    if (!options.force && this.activeEvent && this.activeEvent.actionId === canonical && this.currentState === 'EXECUTING') {
      // Suppress redundant trigger while current gesture is already actively animating
      return null;
    }

    if (options.metadata?.sequenceId) {
      this.activeSequenceId = options.metadata.sequenceId;
    }

    // Check if this is continuous static hand hold without intentional rest/pause
    if (!options.force && options.source === 'vision' && this.lastExecutedSign) {
      const isSameAsLast = this.lastExecutedSign.sign === canonical;
      const elapsedSinceCompletion = now - this.lastExecutedSign.completedAt;

      if (isSameAsLast && elapsedSinceCompletion < this.REPETITION_COOLDOWN_MS) {
        // Continuous static hand hold; suppress infinite replay loop
        return null;
      }
    }

    // 3. Lifecycle: VALIDATED
    this.transitionEventState(event, 'VALIDATED');
    this.processedEventIds.add(eventId);

    // Prune processed cache if growing too large (keep last 200)
    if (this.processedEventIds.size > 200) {
      const it = this.processedEventIds.values();
      for (let i = 0; i < 50; i++) {
        const val = it.next().value;
        if (val) this.processedEventIds.delete(val);
      }
    }

    // 4. Lifecycle: QUEUED
    this.transitionEventState(event, 'QUEUED');
    this.queue.push(event);

    // If IDLE, start execution immediately
    if (this.currentState === 'IDLE' || !this.activeEvent) {
      this.processNextInQueue();
    }

    return event;
  }

  /**
   * Advances the queue and executes the next valid action.
   */
  private processNextInQueue() {
    if (this.queue.length === 0) {
      this.activeEvent = null;
      this.currentState = 'IDLE';
      this.dispatcher?.returnToIdle(0.25);
      return;
    }

    const nextEvent = this.queue.shift()!;
    const now = Date.now();

    // Check for stale events (> MAX_QUEUE_AGE_MS)
    if (now - nextEvent.timestamp > this.MAX_QUEUE_AGE_MS) {
      this.transitionEventState(nextEvent, 'CANCELLED');
      this.processNextInQueue();
      return;
    }

    // 5. Lifecycle: EXECUTING
    this.activeEvent = nextEvent;
    nextEvent.startedAt = now;
    this.transitionEventState(nextEvent, 'EXECUTING');

    if (!this.dispatcher) {
      // Standalone mode / no 3D canvas attached; auto-complete after mock duration
      const mockDuration = 1200;
      setTimeout(() => {
        this.completeActiveEvent();
      }, mockDuration);
      return;
    }

    // Dispatch to 3D AnimationController
    let hasFinished = false;
    const onFinishedCallback = () => {
      if (hasFinished) return;
      hasFinished = true;
      if (this.executionTimeoutTimer) {
        clearTimeout(this.executionTimeoutTimer);
        this.executionTimeoutTimer = null;
      }
      this.completeActiveEvent();
    };

    const played = this.dispatcher.playClip(
      nextEvent.actionId,
      { loop: false, crossFadeDuration: 0.2 },
      onFinishedCallback
    );

    if (!played) {
      // Action failed to play (e.g. unknown track) -> complete gracefully
      this.transitionEventState(nextEvent, 'FAILED');
      this.activeEvent = null;
      this.processNextInQueue();
      return;
    }

    // Fallback safety timer in case animation mixer event does not fire (max 4.5s)
    this.executionTimeoutTimer = setTimeout(() => {
      if (!hasFinished && this.activeEvent === nextEvent) {
        console.warn(`[ActionExecutionManager] Fallback timer triggered for ${nextEvent.actionId}`);
        onFinishedCallback();
      }
    }, 4500);
  }

  /**
   * 6. Lifecycle: COMPLETED → IDLE (or next in queue)
   */
  private completeActiveEvent() {
    if (!this.activeEvent) return;

    const completedEvent = this.activeEvent;
    completedEvent.completedAt = Date.now();
    completedEvent.durationMs = completedEvent.startedAt
      ? completedEvent.completedAt - completedEvent.startedAt
      : undefined;

    this.transitionEventState(completedEvent, 'COMPLETED');
    this.lastExecutedSign = {
      sign: completedEvent.actionId,
      completedAt: completedEvent.completedAt,
      eventId: completedEvent.eventId,
    };

    this.activeEvent = null;

    // Check if more events in queue
    if (this.queue.length > 0) {
      this.processNextInQueue();
    } else {
      // 7. Lifecycle: IDLE
      this.currentState = 'IDLE';
      this.dispatcher?.returnToIdle(0.25);
      if (this.onSequenceComplete) {
        const seqId = this.activeSequenceId || undefined;
        this.activeSequenceId = null;
        try {
          this.onSequenceComplete(seqId);
        } catch (err) {
          console.error('[ActionExecutionManager] onSequenceComplete callback error:', err);
        }
      }
    }
  }

  public getQueuedActions(): string[] {
    return this.queue.map((e) => e.actionId);
  }

  /**
   * Forcefully cancels all pending queued actions and returns immediately to IDLE.
   */
  public resetQueue() {
    if (this.executionTimeoutTimer) {
      clearTimeout(this.executionTimeoutTimer);
      this.executionTimeoutTimer = null;
    }
    while (this.queue.length > 0) {
      const evt = this.queue.shift()!;
      this.transitionEventState(evt, 'CANCELLED');
    }
    if (this.activeEvent) {
      this.transitionEventState(this.activeEvent, 'CANCELLED');
      this.activeEvent = null;
    }
    this.currentState = 'IDLE';
    this.dispatcher?.returnToIdle(0.2);
  }

  /**
   * Notifies subscribers of state transitions.
   */
  private transitionEventState(event: ActionEvent, newState: ActionLifecycleState) {
    const oldState = event.state;
    event.state = newState;
    if (event === this.activeEvent || newState === 'EXECUTING' || newState === 'IDLE') {
      this.currentState = newState;
    }
    this.listeners.forEach((listener) => {
      try {
        listener(event, oldState);
      } catch (err) {
        console.error('[ActionExecutionManager] Listener error:', err);
      }
    });
  }
}
