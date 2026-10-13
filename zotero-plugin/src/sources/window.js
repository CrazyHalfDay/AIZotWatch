// Date windows for source queries. Like the Python sources, only complete past
// days are queried: the window ends yesterday (UTC) and spans `daysBack` days.

const DAY = 86400000;

export function isoDate(ms) {
  return new Date(ms).toISOString().slice(0, 10);
}

export function pastDaysWindow(daysBack, now = Date.now()) {
  const d = new Date(now);
  const todayStart = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate());
  const until = todayStart - DAY;
  const since = until - (Math.max(1, daysBack) - 1) * DAY;
  return { since, until };
}
