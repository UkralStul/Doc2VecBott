from aiogram.dispatcher.filters.state import StatesGroup, State


class message(StatesGroup):
    waiting_for_message = State()

