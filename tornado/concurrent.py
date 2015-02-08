#!/usr/bin/env python
#
# Copyright 2012 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
"""Utilities for working with threads and ``Futures``.

``Futures`` are a pattern for concurrent programming introduced in
Python 3.2 in the `concurrent.futures` package (this package has also
been backported to older versions of Python and can be installed with
``pip install futures``).  Tornado will use `concurrent.futures.Future` if
it is available; otherwise it will use a compatible class defined in this
module.
"""
from __future__ import absolute_import, division, print_function, with_statement

import functools
import platform
import traceback
import sys

from tornado.log import app_log
from tornado.stack_context import ExceptionStackContext, wrap
from tornado.util import raise_exc_info, ArgReplacer

try:
    from concurrent import futures
except ImportError:
    futures = None


# Can the garbage collector handle cycles that include __del__ methods?
# This is true in cpython beginning with version 3.4 (PEP 442).
_GC_CYCLE_FINALIZERS = (platform.python_implementation() == 'CPython' and
                        sys.version_info >= (3, 4))

class ReturnValueIgnoredError(Exception):
    pass

# This class and associated code in the future object is derived
# from the Trollius project, a backport of asyncio to Python 2.x - 3.x
# 该类 和相关的代码 在 future object 被派生
# 从 from the Trollius project, 逆向移植 asyncio to Python 2.x - 3.x

class _TracebackLogger(object):
    """Helper to log a traceback upon destruction if not cleared.

    This solves a nasty problem with Futures and Tasks that have an
    exception set: if nobody asks for the exception, the exception is
    never logged.  This violates the Zen of Python: 'Errors should
    never pass silently.  Unless explicitly silenced.'

    However, we don't want to log the exception as soon as
    set_exception() is called: if the calling code is written
    properly, it will get the exception and handle it properly.  But
    we *do* want to log it if result() or exception() was never called
    -- otherwise developers waste a lot of time wondering why their
    buggy code fails silently.

    An earlier attempt added a __del__() method to the Future class
    itself, but this backfired because the presence of __del__()
    prevents garbage collection from breaking cycles.  A way out of
    this catch-22 is to avoid having a __del__() method on the Future
    class itself, but instead to have a reference to a helper object
    with a __del__() method that logs the traceback, where we ensure
    that the helper object doesn't participate in cycles, and only the
    Future has a reference to it.

    The helper object is added when set_exception() is called.  When
    the Future is collected, and the helper is present, the helper
    object is also collected, and its __del__() method will log the
    traceback.  When the Future's result() or exception() method is
    called (and a helper object is present), it removes the the helper
    object, after calling its clear() method to prevent it from
    logging.

    One downside is that we do a fair amount of work to extract the
    traceback from the exception, even when it is never logged.  It
    would seem cheaper to just store the exception object, but that
    references the traceback, which references stack frames, which may
    reference the Future, which references the _TracebackLogger, and
    then the _TracebackLogger would be included in a cycle, which is
    what we're trying to avoid!  As an optimization, we don't
    immediately format the exception; we only do the work when
    activate() is called, which call is delayed until after all the
    Future's callbacks have run.  Since usually a Future has at least
    one callback (typically set by 'yield From') and usually that
    callback extracts the callback, thereby removing the need to
    format the exception.

    PS. I don't claim credit for this solution.  I first heard of it
    in a discussion about closing files when they are collected.
    """

    __slots__ = ('exc_info', 'formatted_tb')

    def __init__(self, exc_info):
        self.exc_info = exc_info
        self.formatted_tb = None

    def activate(self):
        exc_info = self.exc_info
        if exc_info is not None:
            self.exc_info = None
            self.formatted_tb = traceback.format_exception(*exc_info)

    def clear(self):
        self.exc_info = None
        self.formatted_tb = None

    def __del__(self):
        if self.formatted_tb:
            app_log.error('Future exception was never retrieved: %s',
                          ''.join(self.formatted_tb).rstrip())


class Future(object):
    """Placeholder for an asynchronous result.

    A ``Future`` encapsulates the result of an asynchronous
    operation.  In synchronous applications ``Futures`` are used
    to wait for the result from a thread or process pool; in
    Tornado they are normally used with `.IOLoop.add_future` or by
    yielding them in a `.gen.coroutine`.

    Future 封装异步操作的结果. 在同步应用中 Future 被用于等待一个线程或者进程池返回的结果;
    在 Tornado 里通常用于 IOLoop.add_future  或者在 .gen.coroutine 里被 yielding

    `tornado.concurrent.Future` is similar to
    `concurrent.futures.Future`, but not thread-safe (and therefore
    faster for use with single-threaded event loops).

    tornado.concurrent.Future 与 concurrent.futures.Future` 但是非线程安全
    (因此比使用单线程的 event loops 更快)

    In addition to ``exception`` and ``set_exception``, methods ``exc_info``
    and ``set_exc_info`` are supported to capture tracebacks in Python 2.
    The traceback is automatically available in Python 3, but in the
    Python 2 futures backport this information is discarded.
    This functionality was previously available in a separate class
    ``TracebackFuture``, which is now a deprecated alias for this class.

    除 exception 和 set_exception 之外, methods(方法) exc_info 和 set_exc_info
    在 Python2  用于捕获 traceback. traceback 在 Python3 里是自带的, 而在 Python2 里
    futures 逆向移植 traceback 时, 其信息被丢弃.
    该功能曾在独立的 TracebackFuture类中提供, 现在已经被废弃.

    .. versionchanged:: 4.0
       `tornado.concurrent.Future` is always a thread-unsafe ``Future``
       with support for the ``exc_info`` methods.  Previously it would
       be an alias for the thread-safe `concurrent.futures.Future`
       if that package was available and fall back to the thread-unsafe
       implementation if it was not.

    版本改变: 4.0
    tornado.concurrent.Future 总是非线程安全的 Future, 并支持 exc_info 方法.
    预先地, 如果 concurrent package 有支持. 它是线程安全级别的 concurrent.futures.Future 的别名.
    如果 concurrent 不支持, 则为非线程安全.

    .. versionchanged:: 4.1
       If a `.Future` contains an error but that error is never observed
       (by calling ``result()``, ``exception()``, or ``exc_info()``),
       a stack trace will be logged when the `.Future` is garbage collected.
       This normally indicates an error in the application, but in cases
       where it results in undesired logging it may be necessary to
       suppress the logging by ensuring that the exception is observed:
       ``f.add_done_callback(lambda f: f.exception())``.

    版本改变: 4.1
    如果 一个 .Future  包含 一个 error 但是 error 没有被监控到
    (调用result() exception() exc_info()时),
    在 .Future 被垃圾回收时 一个 stack trace 会被 log.
    通常指的是在 application 的 error, 但有例外, 在它引起的非预料 logging 中,
    可能需要抑制 logging, 通过确认其 exception 被检测到: f.add_done_callback(lambda f: f.exception())
    """
    def __init__(self):
        self._done = False
        self._result = None
        self._exc_info = None

        self._log_traceback = False   # Used for Python >= 3.4
        self._tb_logger = None        # Used for Python <= 3.3

        self._callbacks = []

    def cancel(self):
        """Cancel the operation, if possible.

        Tornado ``Futures`` do not support cancellation, so this method always
        returns False.

        Tornado 的 Futures 不支持 cancellation, 该函数一直返回 False
        """
        return False

    def cancelled(self):
        """Returns True if the operation has been cancelled.

        Tornado ``Futures`` do not support cancellation, so this method
        always returns False.
        """
        return False

    def running(self):
        """Returns True if this operation is currently running."""
        return not self._done

    def done(self):
        """Returns True if the future has finished running."""
        return self._done

    def _clear_tb_log(self):
        self._log_traceback = False
        if self._tb_logger is not None:
            self._tb_logger.clear()
            self._tb_logger = None

    def result(self, timeout=None):
        """If the operation succeeded, return its result.  If it failed,
        re-raise its exception.
        """
        self._clear_tb_log()
        if self._result is not None:
            return self._result
        if self._exc_info is not None:
            raise_exc_info(self._exc_info)
        self._check_done()
        return self._result

    def exception(self, timeout=None):
        """If the operation raised an exception, return the `Exception`
        object.  Otherwise returns None.
        """
        self._clear_tb_log()
        if self._exc_info is not None:
            return self._exc_info[1]
        else:
            self._check_done()
            return None

    def add_done_callback(self, fn):
        """Attaches the given callback to the `Future`.

        关联 给定的 callback 到 Future

        It will be invoked with the `Future` as its argument when the Future
        has finished running and its result is available.  In Tornado
        consider using `.IOLoop.add_future` instead of calling
        `add_done_callback` directly.

        当 Future 执行完成 running 且结果已产生时, fn 作为 Future 的参数被调用.
        在 Tornado 里可以考虑使用 .IOLoop.add_future 替代 直接调用 add_done_callback

        """
        if self._done:
            fn(self)
        else:
            self._callbacks.append(fn)

    def set_result(self, result):
        """Sets the result of a ``Future``.

        It is undefined to call any of the ``set`` methods more than once
        on the same object.
        """
        self._result = result
        self._set_done()

    def set_exception(self, exception):
        """Sets the exception of a ``Future.``"""
        self.set_exc_info(
            (exception.__class__,
             exception,
             getattr(exception, '__traceback__', None)))

    def exc_info(self):
        """Returns a tuple in the same format as `sys.exc_info` or None.

        .. versionadded:: 4.0
        """
        self._clear_tb_log()
        return self._exc_info

    def set_exc_info(self, exc_info):
        """Sets the exception information of a ``Future.``

        Preserves tracebacks on Python 2.

        .. versionadded:: 4.0
        """
        self._exc_info = exc_info
        self._log_traceback = True
        if not _GC_CYCLE_FINALIZERS:
            self._tb_logger = _TracebackLogger(exc_info)

        try:
            self._set_done()
        finally:
            # Activate the logger after all callbacks have had a
            # chance to call result() or exception().
            # 在所有 callback 已经有一个机会 调用 result() 或者 exception() 之后, 激活 logger,
            if self._log_traceback and self._tb_logger is not None:
                self._tb_logger.activate()
        self._exc_info = exc_info

    def _check_done(self):
        if not self._done:
            raise Exception("DummyFuture does not support blocking for results")

    def _set_done(self):
        self._done = True
        for cb in self._callbacks:
            try:
                cb(self)
            except Exception:
                app_log.exception('exception calling callback %r for %r',
                                  cb, self)
        self._callbacks = None

    # On Python 3.3 or older, objects with a destructor part of a reference
    # cycle are never destroyed. It's no longer the case on Python 3.4 thanks to
    # the PEP 442.
    # 在 Python 3.3 或者更老的版本, 带有引用的析构部分的 object 永远不会析构.
    # 但在 Python 3.4 不会存在, 因为在 PEP 442 解决

    if _GC_CYCLE_FINALIZERS:
        def __del__(self):
            if not self._log_traceback:
                # set_exception() was not called, or result() or exception()
                # has consumed the exception
                return

            tb = traceback.format_exception(*self._exc_info)

            app_log.error('Future %r exception was never retrieved: %s',
                          self, ''.join(tb).rstrip())

TracebackFuture = Future

if futures is None:
    FUTURES = Future
else:
    # Python 3.2 之后或者安装了 concurrent库, FUTURES 是一个 tuple, 优先使用 concurrent.futures.Future
    FUTURES = (futures.Future, Future)


def is_future(x):
    return isinstance(x, FUTURES)


class DummyExecutor(object):
    def submit(self, fn, *args, **kwargs):
        future = TracebackFuture()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception:
            future.set_exc_info(sys.exc_info())
        return future

    def shutdown(self, wait=True):
        pass

dummy_executor = DummyExecutor()


def run_on_executor(fn):
    """Decorator to run a synchronous method asynchronously on an executor.

    通过装饰器, 在一个 executor 上异步的方式运行 一个的同步的方法 fn

    The decorated method may be called with a ``callback`` keyword
    argument and returns a future.

    被装饰的 method 被称作 callback 参数的关键词, 并返回一个 future

    This decorator should be used only on methods of objects with attributes
    ``executor`` and ``io_loop``.

    该装饰器应该只使用于拥有 excutor 和 io_loop 属性的 object的 method

    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        callback = kwargs.pop("callback", None)
        future = self.executor.submit(fn, self, *args, **kwargs)
        if callback:
            self.io_loop.add_future(future,
                                    lambda future: callback(future.result()))
        return future
    return wrapper


_NO_RESULT = object()


def return_future(f):
    """Decorator to make a function that returns via callback return a
    `Future`.

    装饰器, 通过 callback, 使 function 返回 一个 Future 　

    The wrapped function should take a ``callback`` keyword argument
    and invoke it with one argument when it has finished.  To signal failure,
    the function can simply raise an exception (which will be
    captured by the `.StackContext` and passed along to the ``Future``).

    包装的 function 必须有一个 callback 关键字参数 而且在 function 结束时
    传入一个参数调用 callback. 为了通知 failure, 该 function 简单地 raise
    一个 exception (它会被 .StackContext 捕获, 并且被传递到 Future ).

    From the caller's perspective, the callback argument is optional.
    If one is given, it will be invoked when the function is complete
    with `Future.result()` as an argument.  If the function fails, the
    callback will not be run and an exception will be raised into the
    surrounding `.StackContext`.

    从 caller 的角度看, callback 参数是可选的. 如果给定, callback 会在 function
    完成时执行 Future.result() 作为参数被调用. 如果 function 执行失败, callback
    不会执行, 而且 exception 被 raised 进 .StackContext

    If no callback is given, the caller should use the ``Future`` to
    wait for the function to complete (perhaps by yielding it in a
    `.gen.engine` function, or passing it to `.IOLoop.add_future`).

    如果 没有 callback 指定, caller 调用者会使用 Future 等待 function 完成
    (可能被 yielding 在 .gen.engine function, 或者传递到 .IOLoop.add_future)

    Usage::

        @return_future
        def future_func(arg1, arg2, callback):
            # Do stuff (possibly asynchronous)
            callback(result)

        @gen.engine
        def caller(callback):
            yield future_func(arg1, arg2)
            callback()

    Note that ``@return_future`` and ``@gen.engine`` can be applied to the
    same function, provided ``@return_future`` appears first.  However,
    consider using ``@gen.coroutine`` instead of this combination.

    注意: @return_future 和 @gen.engine 可以应用于相同的 function,
    先使用 @return_future. 然而, 可以考虑使用 @gen.coroutine 替代以上组合.

    """
    replacer = ArgReplacer(f, 'callback')

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        future = TracebackFuture()
        callback, args, kwargs = replacer.replace(
            lambda value=_NO_RESULT: future.set_result(value),
            args, kwargs)

        def handle_error(typ, value, tb):
            future.set_exc_info((typ, value, tb))
            return True
        exc_info = None
        with ExceptionStackContext(handle_error):
            try:
                result = f(*args, **kwargs)
                if result is not None:
                    raise ReturnValueIgnoredError(
                        "@return_future should not be used with functions "
                        "that return values")
            except:
                exc_info = sys.exc_info()
                raise

        # with 语句抛出的异常不会直接继续抛出, 因为 handle_error 一直返回 True,
        # ExceptionStackContext 类里的 __exit__ 函数处理异常时, 调用的是 handle_error
        if exc_info is not None:
            # 这里才真正处理异常
            # If the initial synchronous part of f() raised an exception,
            # go ahead and raise it to the caller directly without waiting
            # for them to inspect the Future.
            # 4.1 继续执行 future.result()
            future.result()

        # If the caller passed in a callback, schedule it to be called
        # when the future resolves.  It is important that this happens
        # just before we return the future, or else we risk confusing
        # stack contexts with multiple exceptions (one here with the
        # immediate exception, and again when the future resolves and
        # the callback triggers its exception by calling future.result()).

        """
        如果 caller 传递进一个 callback, 另它在 future 解析时被调用.
        在我们返回 future 之前, 这是非常重要, 否则我们冒险地将多个 exception
        混淆了 stack context(这里一个紧急的 exception, 而 future 解析时又一个, 且
        callback 在调用 future.result() 时出发了它自己的 exception )
        """
        if callback is not None:
            def run_callback(future):
                result = future.result()
                if result is _NO_RESULT:
                    callback()
                else:
                    callback(future.result())
            future.add_done_callback(wrap(run_callback))
        return future
    return wrapper


def chain_future(a, b):
    """Chain two futures together so that when one completes, so does the other.

    将两个 future 链接一块, 因此其中一个 完成(completed) 时, 另一个也完成(completed).

    The result (success or failure) of ``a`` will be copied to ``b``, unless
    ``b`` has already been completed or cancelled by the time ``a`` finishes.

    a 的 result (成功或失败) 会拷贝到 b, 除非 b 已经完成或者 a finish 时, 取消 b.

    """
    def copy(future):
        assert future is a
        if b.done():
            return
        if (isinstance(a, TracebackFuture) and isinstance(b, TracebackFuture)
                and a.exc_info() is not None):
            b.set_exc_info(a.exc_info())
        elif a.exception() is not None:
            b.set_exception(a.exception())
        else:
            b.set_result(a.result())
    a.add_done_callback(copy)
