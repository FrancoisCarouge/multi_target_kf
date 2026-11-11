// The log-storm suppression of the "dt <= 0" warnings, shared by the motion models.
#ifndef DT_WARNING_H
#define DT_WARNING_H

#include <cinttypes>
#include <cstdint>
#include <cstdio>

/* ---------------------------------------------------------------------
 * Log-storm suppression for the "dt <= 0" warnings.
 *
 * Those warnings sit on a PER-ITERATION path, and dt <= 0 is a latching
 * condition: once a model is handed a clock that stops advancing (a paused
 * or stepped time source, a replayed log that wraps, a sensor republishing
 * a stamp) every subsequent call takes the same branch. The warning was
 * unconditional, so a single stuck clock produced a continuous stream of
 * identical lines -- measured at 50-100 lines/second on a 30 Hz pipeline,
 * roughly a gigabyte of logs a day, which then evicts every other record.
 *
 * Simply deleting the warning, or hiding it behind debug_, would trade a
 * flood for silence: dt <= 0 means the filter is not advancing, and that
 * is worth knowing. So instead: report the FIRST occurrence of a run in
 * full, count the rest, print a periodic summary carrying that count, and
 * report once when the condition clears.
 *
 * This is logging only. Nothing below is read by any model, and none of it
 * enters a state vector, covariance, sigma point or return value.
 * ------------------------------------------------------------------- */
struct DtWarnState {
    uint64_t count;            /* occurrences in the current run */
    bool active;               /* currently inside a run of dt <= 0 */
    DtWarnState() : count(0), active(false) {}
};

/* Call where dt <= 0 is detected. `where` identifies the call site. */
inline void reportNonPositiveDt(DtWarnState& st, const char* where, double dt) {
    const uint64_t log_every = 10000;
    ++st.count;
    if (!st.active) {
        st.active = true;
        printf("WARN [%s] dt = %f <= 0. Returning same state. "
               "Repeats will be counted, not printed.\n", where, dt);
    } else if (st.count % log_every == 0) {
        printf("WARN [%s] dt still <= 0 (last dt = %f) after %" PRIu64
               " consecutive occurrences. The time source is not advancing.\n",
               where, dt, st.count);
    }
}

/* Call on the dt > 0 path, to close an open run and report what it hid. */
inline void clearNonPositiveDt(DtWarnState& st, const char* where) {
    if (st.active) {
        printf("WARN [%s] dt > 0 again after %" PRIu64
               " consecutive non-positive dt.\n", where, st.count);
        st.active = false;
        st.count = 0;
    }
}

#endif // DT_WARNING_H
