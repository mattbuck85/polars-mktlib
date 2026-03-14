Scheduling
==========

Exchange calendars — sessions, minutes, and trading index.

Included in the base ``pip install mktlib`` — no extra dependencies.

.. autofunction:: mktlib.scheduling.get_calendar

ExchangeCalendar
----------------

Core
~~~~

.. automethod:: mktlib.scheduling.ExchangeCalendar.valid_days
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.schedule
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.is_session
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.get_schedule
   :no-index:

Session Navigation
~~~~~~~~~~~~~~~~~~

.. automethod:: mktlib.scheduling.ExchangeCalendar.next_session
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.previous_session
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.session_offset
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.date_to_session
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.sessions_in_range
   :no-index:

Minute Queries
~~~~~~~~~~~~~~

.. automethod:: mktlib.scheduling.ExchangeCalendar.is_open_on_minute
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.next_open
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.next_close
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.previous_open
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.previous_close
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.minute_to_session
   :no-index:

Trading Helpers
~~~~~~~~~~~~~~~

.. automethod:: mktlib.scheduling.ExchangeCalendar.filter_market_hours
   :no-index:
.. automethod:: mktlib.scheduling.ExchangeCalendar.trading_index
   :no-index:

Full Class Reference
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: mktlib.scheduling.ExchangeCalendar
   :members:
   :show-inheritance:

ExchangeCalendarWithBreaks
--------------------------

``ExchangeCalendarWithBreaks`` extends :class:`ExchangeCalendar` with lunch-break
support (JPX, HKEX, SSE). The ``BreakMixin`` overrides four methods:

- :meth:`~ExchangeCalendar.schedule` adds ``break_start`` / ``break_end`` columns
- :meth:`~ExchangeCalendar.is_open_on_minute` returns ``False`` during the break window
- :meth:`~ExchangeCalendar.trading_index` generates separate morning and afternoon ranges
- :meth:`~ExchangeCalendar.filter_market_hours` excludes lunch bars

.. autoclass:: mktlib.scheduling.ExchangeCalendarWithBreaks
   :members:
   :show-inheritance:

Custom Calendar Example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datetime import date, time
   from mktlib.scheduling import ExchangeCalendarWithBreaks, register_exchange
   from mktlib.scheduling.rules import HolidayRule

   cal = ExchangeCalendarWithBreaks(
       name="XSHG",
       timezone="Asia/Shanghai",
       open_time=time(9, 30),
       close_time=time(15, 0),
       break_start=time(11, 30),
       break_end=time(13, 0),
       holidays=[
           HolidayRule("New Year's Day", month=1, day=1),
           HolidayRule("National Day", month=10, day=1),
       ],
   )

   register_exchange("XSHG", lambda: cal, aliases=["Shanghai", "SSE"])

Holiday Rules
-------------

.. autoclass:: mktlib.scheduling.rules.HolidayRule
   :members:

.. autoclass:: mktlib.scheduling.rules.AdhocClosure
   :members:

.. autoclass:: mktlib.scheduling.rules.EarlyClose
   :members:
