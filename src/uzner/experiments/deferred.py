"""Безопасная зависимость GPU-очереди от конкретных Linux-процессов."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessIdentity:
    """PID и время старта защищают очередь от повторного использования PID."""

    pid: int
    start_ticks: str

    def alive(self, proc_root: Path = Path("/proc")) -> bool:
        """Проверяет именно исходный процесс, не блокируясь на zombie."""
        try:
            raw = (proc_root / str(self.pid) / "stat").read_text()
        except FileNotFoundError:
            return False
        fields = raw.rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == self.start_ticks

    @classmethod
    def parse(cls, value: str) -> "ProcessIdentity":
        """Разбирает явно переданную пару PID:START_TICKS."""
        pid, ticks = value.split(":")
        if not pid.isdigit() or int(pid) <= 0 or not ticks.isdigit():
            raise ValueError("Нужна положительная пара PID:START_TICKS")
        return cls(int(pid), ticks)
