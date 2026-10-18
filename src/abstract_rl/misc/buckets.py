"""Power-of-two prefixes for buffer-wide MPO kernels.

A jitted scan whose length is the exact fill retraces every epoch. Rounding the fill up to a
power of two, and using the allocation itself once that power would pass it, bounds the number
of compiles by the log of the capacity. The steady state, a full buffer, is one of those shapes.
"""


def active_width(size: int, capacity: int) -> int:
    """Smallest power of two that covers ``size``, never above ``capacity``.

    Args:
        size: Number of stored rows.
        capacity: Allocated rows.

    Returns:
        The prefix length to compile. ``0`` when nothing is stored, and ``capacity`` when the
        buffer is full or the next power of two would be larger than the allocation.
    """
    if size <= 0 or capacity <= 0:
        return 0
    if size >= capacity:
        return capacity
    width = 1
    while width < size:
        width *= 2
    if width > capacity:
        return capacity
    return width


def chunk_count(size: int, capacity: int, chunk: int) -> int:
    """How many ``chunk``-sized blocks cover :func:`active_width`.

    Args:
        size: Number of stored rows.
        capacity: Allocated rows.
        chunk: Block length of the scan.

    Returns:
        The static scan length. At least one when ``size`` is positive.
    """
    width = active_width(size, capacity)
    if width <= 0:
        return 0
    return (width + chunk - 1) // chunk


def slot_width(n_items: int, full: int, quantum: int) -> int:
    """Power-of-two slot count covering ``n_items``, capped at the full-buffer slot count.

    Args:
        n_items: Episodes currently stored.
        full: Slot count used when the allocation is the scan length.
        quantum: The scan consumes this many slots at a time, so the width is a multiple of it.

    Returns:
        The static slot count. ``full`` once the stored episodes no longer fit in a smaller bucket.
    """
    if n_items <= 0 or full <= 0:
        return 0
    if n_items >= full:
        return full
    width = max(quantum, 1)
    while width < n_items:
        width *= 2
    if width > full:
        return full
    return width
