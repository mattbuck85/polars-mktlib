# mktlib

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Polars-native financial market toolkit. Zero pandas dependency.

## Installation

```bash
pip install mktlib
```

## Subpackages

| Package | Purpose |
|-|-|
| `mktlib.scheduling` | Exchange calendars, trading schedules, holiday rules |

## Usage

```python
from mktlib.scheduling import get_calendar

cal = get_calendar("XNYS")
schedule = cal.schedule("2024-01-02", "2024-12-31")
trading_days = cal.valid_days("2024-01-02", "2024-12-31")
```

## Development

```bash
pip install -e ".[dev]"
pytest
pyright src/mktlib
```

## License

MIT
