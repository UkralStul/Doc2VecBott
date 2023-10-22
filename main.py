from gensim.models import Doc2Vec
import aiogram.utils.markdown as md
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor
from aiogram.dispatcher import FSMContext
import FSM_classes
model = Doc2Vec.load("doc2vec_model")

def tokenize_text(text):
    return text.lower().split()


def classify_phrase(phrase):
    vec = model.infer_vector(tokenize_text(phrase))
    return model.dv.most_similar([vec], topn=1)[0][0]

bot = Bot(token="6349311026:AAE97ZTAl67uZ1xzpD773spYgaHF843mxYU")
dp = Dispatcher(bot, storage=MemoryStorage())

@dp.message_handler(commands="start")
async def cmd_start(message: types.Message):
    await message.reply("Привет!")
    await FSM_classes.message.waiting_for_message.set()

@dp.message_handler(state= FSM_classes.message.waiting_for_message)
async def get_message(message: types.Message, state: FSMContext):
    phrase_class = classify_phrase(message.text)
    if phrase_class == 'ФразаПриветствие':
        await bot.send_message(chat_id=message.from_user.id, text='Привет!')
    elif phrase_class == 'ФразаПогода':
        await bot.send_message(chat_id=message.from_user.id, text='Сегодня отличная погодка!')
    elif phrase_class == 'ФразаКто':
        await bot.send_message(chat_id=message.from_user.id, text='Я бот!')
    elif phrase_class == 'ФразаПрощание':
        await bot.send_message(chat_id=message.from_user.id, text='До новых встреч!')
    await FSM_classes.message.waiting_for_message.set()

if __name__ == "__main__":
    print("Starting bot...")
    executor.start_polling(dp, skip_updates=True)