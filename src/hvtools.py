# This file is part of hvsrprocpy, a Python package for horizontal-to-vertical
# spectral ratio processing.

# Copyright (c) 2024 Francisco Javier Ornelas (jornela1@g.ucla.edu)

#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.

#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.

"""Function definitions for hvtools."""

from hvsrprocpy import hvproc
from hvsrprocpy import tsproc
from hvsrprocpy import hvmetaproc

__all__ = ['_ts_tool', '_hv_tool', '_meta_tool', 'combined_hv_tools']


def _ts_tool(h1, h2, v, dt, time, directory, output_dir, **kwargs):
    ts_obj = tsproc.TsProc(h1=h1, h2=h2, v=v, dt=dt, time=time, directory=directory,
                           output_dir=output_dir)

    ts_defaults, win_results = ts_obj.ts_proc(**kwargs)

    return ts_defaults, win_results


def _hv_tool(h1_wins, h2_wins, v_wins, dt, win_indicies, output_dir, **kwargs):
    hvsr = hvproc.HvsrProc(h1_wins=h1_wins, h2_wins=h2_wins,
                           v_wins=v_wins, dt=dt,
                           idx_select_ts=win_indicies, output_dir=output_dir)

    fd_defaults, fd_select = hvsr.hv_proc(**kwargs)

    return fd_defaults, fd_select


def _meta_tool(ts_defaults, fd_defaults, win_results, fd_select, output_dir, **kwargs):
    meta = hvmetaproc.HvMetaProc(ts_def_list=ts_defaults, fd_def_list=fd_defaults, ts_list=win_results,
                                 fd_list=fd_select, output_dir=output_dir)

    meta_output = meta.gen_meta(**kwargs)

    return meta_output


def combined_hv_tools(h1, h2, v, dt, time, directory, output_dir, **kwargs):
    ts_defaults, win_results = _ts_tool(h1=h1, h2=h2, v=v, dt=dt, time=time, directory=directory, output_dir=output_dir,
                                        **kwargs)

    fd_defaults, fd_select = _hv_tool(h1_wins=win_results['h1_wins'], h2_wins=win_results['h2_wins'],
                                      v_wins=win_results['v_wins'], dt=dt,
                                      win_indicies=win_results['idx_select'], output_dir=output_dir, **kwargs)

    meta_out = _meta_tool(ts_defaults=ts_defaults, fd_defaults=fd_defaults, win_results=win_results,
                          fd_select=fd_select, output_dir=output_dir, **kwargs)

    return ts_defaults, win_results, fd_defaults, fd_select, meta_out
