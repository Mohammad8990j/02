# Progress Log - Trading Platform
**Project ID**: TRADING_PLATFORM_2025

## Completed (تا 2 مه 2025)
- **اتصال به MetaTrader 5**:
  - دریافت داده‌های قیمتی OHLC برای نمادها (مانند EURUSD، EURUSD_o) در تایم‌فریم‌های مختلف (M1، M5، M15، H1).
- **اندیکاتورهای تکنیکال**:
  - پیاده‌سازی SMA (دوره‌های 10 و 20) و RSI (دوره 14).
  - افزودن MACD (دوره‌های 12، 26، 9)، Bollinger Bands (دوره 20، انحراف 2)، و Stochastic (دوره‌های 14، 3، 3).
  - افزودن MFI (دوره 14) برای شناسایی شرایط اشباع خرید/فروش.
  - محاسبه سطوح حمایت/مقاومت با استفاده از نقاط پیوت (Pivot Points).
- **تولید سیگنال**:
  - استراتژی تقاطع SMA (خرید/فروش).
  - استراتژی‌های اندیکاتوری: تقاطع MACD، شکست Bollinger Bands، سطوح اشباع Stochastic، و اشباع MFI.
  - استراتژی ترکیبی: ترکیب سطوح حمایت/مقاومت، افزایش/کاهش محسوس حجم، و تأیید MFI.
- **بک‌اند**:
  - پیاده‌سازی API با FastAPI برای دریافت داده‌های OHLC و سیگنال‌ها.
  - افزودن سیستم لاگینگ دقیق با استفاده از ماژول `logging`:
    - ذخیره لاگ‌ها در فایل `trading_platform.log` و نمایش در کنسول.
    - ثبت رویدادهای کلیدی (اتصال به MT5، دریافت داده‌ها، محاسبه اندیکاتورها، تولید سیگنال‌ها، و خطاها).
    - پشتیبانی از سطوح لاگ (INFO، DEBUG، ERROR، WARNING).
  - رفع خطای مربوط به ستون `volume`:
    - مدیریت نمادهایی که داده‌های حجم ارائه نمی‌دهند با تنظیم پیش‌فرض مقدار صفر.
    - افزودن لاگ‌های دقیق‌تر برای نمایش ساختار داده‌های دریافت‌شده.
  - بهبود مدیریت حجم:
    - استفاده از `tick_volume` به‌جای `volume` برای محاسبه تغییرات حجم در استراتژی ترکیبی.
    - پشتیبانی از نمادهای بدون حجم با نادیده گرفتن شرط حجم در استراتژی ترکیبی و کاهش Confidence.
  - رفع مشکل CORS:
    - افزودن `CORSMiddleware` به FastAPI برای اجازه دادن به درخواست‌های کراس‌دامین.
    - تنظیم هدرهای `Access-Control-Allow-Origin` برای پشتیبانی از همه منشأها (برای توسعه).
  - افزودن اندیکاتورها به داده‌های OHLC:
    - تغییر endpoint `/ohlc/{symbol}/{timeframe}` برای برگرداندن SMA (دوره‌های 10 و 20)، Bollinger Bands (دوره 20، انحراف 2)، و RSI (دوره 14).
  - بهبود مدیریت خطاها برای رفع خطای 500:
    - افزودن بررسی‌های دقیق‌تر برای داده‌های OHLC و اندیکاتورها.
    - بهبود لاگ‌ها برای شناسایی دلیل خطاهای سرور (مثل مشکلات MT5 یا محاسبات اندیکاتورها).
  - رفع خطای `ValueError: Out of range float values are not JSON compliant: nan`:
    - جایگزینی مقادیر `NaN` با `null` در داده‌های OHLC و اندیکاتورها برای سریال‌سازی صحیح JSON.
    - افزودن لاگ‌های دقیق‌تر برای شناسایی ستون‌ها و ردیف‌های دارای `NaN`.
  - افزودن قابلیت تنظیم اندیکاتورها:
    - آپدیت endpoint `/ohlc/{symbol}/{timeframe}` برای پذیرش پارامترهای دوره (sma_fast_period، sma_slow_period، bb_period، bb_deviation، rsi_period) از طریق query parameters.
    - تغییر تابع `calculate_indicators` برای استفاده از دوره‌های ارسالی از فرانت‌اند.
    - افزودن لاگ‌های مربوط به پارامترهای ورودی برای دیباگ.
- **فرانت‌اند**:
  - رابط کاربری اولیه با React، Tailwind CSS، و Chart.js برای نمایش سیگنال‌ها و نمودار قیمتی.
  - آپدیت رابط کاربری برای نمایش سیگنال‌های جدید (MFI و استراتژی ترکیبی).
  - افزودن جدول تاریخچه سیگنال‌ها با جزئیات (نماد، تایم‌فریم، نوع سیگنال، قیمت، Confidence، اندیکاتور).
  - بهبود طراحی با تم تیره و استایل‌های واکنش‌گرا.
  - رفع مشکل نمایش چارت:
    - بررسی و فیلتر داده‌های OHLC برای حذف مقادیر نامعتبر.
    - افزودن پیام‌های "Loading..." و "No data available" برای حالات بدون داده.
    - بهبود مدیریت Chart.js با استفاده از useRef برای جلوگیری از رندرهای تکراری.
    - افزودن لاگ‌های کنسول برای دیباگ مشکلات چارت.
  - رفع مشکل پیام "No data available for chart":
    - بهبود فیلتر داده‌ها برای سازگاری با فرمت‌های مختلف داده‌های OHLC.
    - افزودن پیام‌های دیباگ دقیق‌تر برای شناسایی دلیل فیلتر شدن داده‌ها.
    - لاگ کردن داده‌های خام و معتبر برای ردیابی مشکلات.
  - رفع خطای "Canvas element for chart not found":
    - اطمینان از رندر همیشگی عنصر `<canvas>` در DOM، حتی بدون داده.
    - استفاده از `useRef` برای دسترسی مستقیم به `<canvas>`.
    - افزودن لاگ‌های کنسول برای بررسی وضعیت رندر `<canvas>` و Chart.js.
    - بهبود پیام‌های خطا در UI برای نمایش دلیل مشکلات چارت.
  - رفع خطای CORS و `long_message`:
    - اصلاح کد برای رفع خطای `ReferenceError: long_message is not defined`.
    - بهبود پیام‌های خطا برای نمایش جزئیات مشکلات شبکه (مثل CORS).
  - نمایش اندیکاتورها روی چارت:
    - افزودن امکان نمایش SMA (دوره‌های 10 و 20) و Bollinger Bands (دوره 20، انحراف 2) روی نمودار قیمتی.
    - افزودن چک‌باکس برای فعال/غیرفعال کردن نمایش این اندیکاتورها در UI.
    - رندر خطوط اندیکاتورها با رنگ‌های متمایز در Chart.js.
  - بهبود لاگ‌ها و مدیریت خطاها:
    - افزودن لاگ‌های دقیق‌تر برای درخواست‌های شبکه و داده‌های دریافتی.
    - بهبود پیام‌های خطا در UI برای نمایش جزئیات خطاهای سرور (مثل خطای 500).
  - افزودن قابلیت تنظیم اندیکاتورها:
    - افزودن فیلدهای ورودی برای تنظیم دوره‌های SMA (Fast و Slow)، Bollinger Bands (دوره و انحراف)، و RSI (دوره) در UI.
    - آپدیت درخواست‌های OHLC برای ارسال پارامترهای دوره به بک‌اند.
    - بهبود UI برای نمایش واضح‌تر تنظیمات اندیکاتورها.
  - نمایش RSI روی چارت:
    - افزودن پنل جداگانه زیر چارت قیمت برای نمایش RSI به‌صورت نمودار خطی.
    - افزودن خطوط افقی برای سطوح اشباع خرید (70) و اشباع فروش (30) با استفاده از پلاگین annotation در Chart.js.
    - افزودن چک‌باکس برای فعال/غیرفعال کردن نمایش RSI.
    - پشتیبانی از تنظیم دوره RSI از UI.

## Next Steps
1. **نمایش اندیکاتورهای بیشتر**:
   - افزودن امکان نمایش MACD و Stochastic روی چارت یا در پنل‌های جداگانه.
   - افزودن فیلدهای تنظیم برای این اندیکاتورها در UI.
2. **استراتژی‌های معاملاتی پیشرفته**:
   - پیاده‌سازی استراتژی پرایس اکشن (شناسایی الگوهای کندلی مثل Pin Bar، Engulfing).
   - افزودن استراتژی ضدروندی مبتنی بر اندیکاتورها (مانند RSI + Stochastic).
3. **هوش مصنوعی**:
   - ادغام مدل‌های یادگیری ماشین (Random Forest، Gradient Boosting) برای تحلیل داده‌ها.
   - استفاده از LSTM برای پیش‌بینی سری‌های زمانی.
4. **مدیریت ریسک**:
   - محاسبه خودکار حد سود (Take Profit) و حد ضرر (Stop Loss) بر اساس نسبت ریسک به ریوارد.
5. **شبیه‌سازی و بک‌تست**:
   - پیاده‌سازی قابلیت بک‌تست برای ارزیابی استراتژی‌ها با داده‌های تاریخی MT5.
6. **بهبود رابط کاربری**:
   - افزودن داشبورد پیشرفته با فیلترهای تاریخچه سیگنال‌ها (بر اساس نماد، تایم‌فریم، یا اندیکاتور).

## Notes
- برای ادامه پروژه، لطفاً از **Project ID: TRADING_PLATFORM_2025** استفاده کنید.
- آرتیفکت‌های کلیدی:
  - Backend: `c16fdcb5-e73b-4156-aea8-1922b9ec04da`
  - Frontend: `2c38d17c-15da-40ad-848c-5f7e6e146415`
  - Progress Log: `05646011-e26b-47fc-9a00-9d8a9db68f08`
- پیشنهاد: کدهای پروژه را در یک مخزن Git ذخیره کنید و لینک آن را به اشتراک بگذارید.
- فایل لاگ (`trading_platform.log`) در دایرکتوری اجرایی برنامه ذخیره می‌شود و برای دیباگ بک‌اند مفید است.
- برای دیباگ مشکلات فرانت‌اند، کنسول مرورگر (F12 > Console) را بررسی کنید و لاگ‌های مربوط به رندر `<canvas>` و داده‌های OHLC را به اشتراک بگذارید.
- برای جلوگیری از مشکلات CORS، فرانت‌اند را از طریق یک سرور محلی (مثل `http-server`) اجرا کنید.