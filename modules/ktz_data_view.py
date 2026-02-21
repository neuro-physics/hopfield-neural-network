import os
import numpy
import pandas
import argparse
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections.abc

def animate_data(xData,Lx,Ly,p=1,anim_interval=20,show_time=True,use_blit=True,theta=0.0,repeat=False,cmap='plasma',figsize=5,ax_title=''):
    """
    xData[i,t] -> data of neuron i at time t
    """
    # pre-processing of data before displaying animation
    xData[xData<theta] = -1.0
    xData              = xData**p
    tTotal             = xData.shape[1]

    assert xData.shape[0]==(Lx*Ly),'The number of neurons (rows in xData) must equal Lx*Ly to fit in a square lattice'

    fig,ax    = plt.subplots(nrows=1,ncols=1,figsize=(figsize,figsize))
    im        = ax.imshow(xData[:,0].reshape((Lx,Ly)),cmap=cmap,extent=[0.5,Lx+0.5,0.5,Ly+0.5])
    im.set_clim((-1.0,1.0))
    if show_time:
        timestamp = ax.text(0.5,0.95, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=ax.transAxes, ha="center")
    ax.set_xticks(numpy.arange(1,Lx+1))
    ax.set_yticks(numpy.arange(1,Ly+1))
    if len(ax_title):
        ax.set_title(ax_title)

    fig.tight_layout()

    def animate(t):
        #print(t,' and ',d.xData.shape[1])
        if (t>=tTotal):
            print('finished animation...')
            #anim.event_source.stop()
            anim.pause()
            if show_time:
                return im,timestamp,
            return im,
        im.set_array(xData[:,t].reshape((Lx,Ly)))
        if show_time:
            timestamp.set_text('t=%d'%t)
            return im,timestamp,
        return im,
    anim = animation.FuncAnimation(fig,animate,frames=tTotal,blit=use_blit,interval=anim_interval,repeat=repeat)
    return anim

def main():
    #sys.argv += ['ktz400_sim.mat']

    #sys.argv = 'ktz_data_view.py processo_poisson_N100_r0.001_T200.mat -save -cmap gray'.split(' ')

    parser = argparse.ArgumentParser(description='Views KTz.exe simulation data stored in either mat or dat files. The file needs to have the xData variable, or have first column=time, remaining columns->x time series.')
    parser.add_argument('ktzdatafile' , nargs='+', metavar='KTZ_DATA', type=str, default=[''] , help='a single file gerated with wData=Yes from KTz.exe')
    parser.add_argument('-cmap'       , required=False, nargs=1, metavar='COLORMAP', type=str   , default=['plasma'] ,choices=plt.colormaps(), help='matplotlib colormap')
    parser.add_argument('-savefmt'    , required=False, nargs=1, metavar='FMT'     , type=str   , default=['mp4']    ,choices=['mp4','gif']  , help='output format for animation')
    parser.add_argument('-figsize'    , required=False, nargs=1, metavar='float_in', type=float , default=[5.0]      ,help='lateral size of figure')
    parser.add_argument('-pow'        , required=False, nargs=1, metavar='float'   , type=float , default=[3.0]      ,help='the color corresponds to x**pow to highlight nuances')
    parser.add_argument('-xthreshold' , required=False, nargs=1, metavar='float'   , type=float , default=[0.0]      ,help='membrane potential threshold (helps visualizing only spikes)')
    parser.add_argument('-interval'   , required=False, nargs=1, metavar='int_ms'  , type=int   , default=[20]       ,help='interval in ms between consecutive frames')
    parser.add_argument('-repeat'     , required=False, action='store_true', default=False, help='if set, repeats the animation indefinitely')
    parser.add_argument('-hidetime'   , required=False, action='store_true', default=False, help='if set, hides time')
    parser.add_argument('-noblit'     , required=False, action='store_true', default=False, help='if set, avoids using blit (may slow the animation, but makes it more precise)')
    parser.add_argument('-save'       , required=False, action='store_true', default=False, help='if set, saves animation')
    args = parser.parse_args()

    try:
        d         = structtype(**scipy.io.loadmat(args.ktzdatafile[0],squeeze_me=True,struct_as_record=False))
    except ValueError:
        d         = load_ktz_data_txt_file(args.ktzdatafile[0])
    p             = args.pow[0]
    anim_interval = args.interval[0]
    show_time     = not args.hidetime
    use_blit      = not args.noblit
    theta         = args.xthreshold[0]
    
    assert d.IsField('xData'), f'The variable xData must be present in the {args.ktzdatafile[0]}'


    """
    ############################
    ############################
    ############################
    ############################
    ############################ Creating and saving animation
    ############################
    ############################
    ############################
    ############################
    """
    anim = animate_data(d.xData,d.Network_Param.Lx,d.Network_Param.Ly,p,anim_interval,show_time,use_blit,theta,repeat=args.repeat,cmap=args.cmap[0],figsize=args.figsize[0],ax_title=args.ktzdatafile[0])

    if args.save:
        out_file_name = check_and_get_filename(os.path.splitext(args.ktzdatafile[0])[0] + f'_{args.cmap[0]}.'+args.savefmt[0])
        print('saving ... ', out_file_name)
        anim.save(out_file_name)

    plt.show()

def load_ktz_data_txt_file(fileName):
    data  = pandas.read_csv(fileName,sep='\t',dtype=float,skipinitialspace=True,skip_blank_lines=True,comment='#').to_numpy()
    Lx,Ly = get_divisors(data.shape[1]-1)
    d     = structtype(xData=data[:,1:].T,Network_Param=structtype(Lx=Lx,Ly=Ly),time=data[:,0])
    return d

def get_divisors(N):
    sN = numpy.sqrt(N)
    if int(numpy.floor(sN))==sN:
        # number has integer sqrt
        return int(sN),int(sN)
    n_list = []
    for n1 in range(int(N)-1,0,-1):
        for n2 in range(int(numpy.floor(sN)),0,-1):
            if ((n1 * n2) == int(N)):
                n_list.append((n1,n2))
    if len(n_list)==0:
        return 1,int(N)
    return n_list[numpy.argmin(numpy.abs(numpy.diff(numpy.array(n_list),axis=1)))]

def check_and_get_filename(fn):
    n = 0
    if os.path.exists(fn):
        old_fn,ext = os.path.splitext(fn)
        while True:
            n += 1
            new_fn = old_fn + '_' + str(n) + ext
            if not os.path.exists(new_fn):
                break
        return new_fn
    else:
        return fn

class structtype(collections.abc.MutableMapping):
    def __init__(self,struct_fields=None,field_values=None,**kwargs):
        if not(type(struct_fields) is type(None)):
            #assert not(type(values) is type(None)),"if you provide field names, you must provide field values"
            if not self._is_iterable(struct_fields):
                struct_fields = [struct_fields]
                field_values = [field_values]
            kwargs.update({f:v for f,v in zip(struct_fields,field_values)})
        self.Set(**kwargs)
    def Set(self,**kwargs):
        self.__dict__.update(kwargs)
        return self
    def SetAttr(self,field,value):
        if not self._is_iterable(field):
            field = [field]
            value = [value]
        self.__dict__.update({f:v for f,v in zip(field,value)})
        return self
    def GetFields(self):
        return '; '.join([ k for k in self.__dict__.keys() if (k[0:2] != '__') and (k[-2:] != '__') ])
        #return self.__dict__.keys()
    def IsField(self,field):
        return field in self.__dict__.keys()
    def RemoveField(self,field):
        return self.__dict__.pop(field,None)
    def RemoveFields(self,*fields):
        r = []
        for k in fields:
            r.append(self.__dict__.pop(k,None))
        return r
    def KeepFields(self,*fields):
        keys = list(self.__dict__.keys())
        for k in keys:
            if not (k in fields):
                self.__dict__.pop(k,None)
    def __setitem__(self,label,value):
        self.__dict__[label] = value
    def __getitem__(self,label):
        return self.__dict__[label]
    def __repr__(self):
        type_name = type(self).__name__
        arg_strings = []
        star_args = {}
        for arg in self._get_args():
            arg_strings.append(repr(arg))
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type_name, ', '.join(arg_strings))
    def _get_kwargs(self):
        return sorted(self.__dict__.items())
    def _get_args(self):
        return []
    def _is_iterable(self,obj):
        return (type(obj) is list) or (type(obj) is tuple)
    def __delitem__(self,*args):
        self.__dict__.__delitem__(*args)
    def __len__(self):
        return self.__dict__.__len__()
    def __iter__(self):
        return iter(self.__dict__)

if __name__ == '__main__':
    main()
