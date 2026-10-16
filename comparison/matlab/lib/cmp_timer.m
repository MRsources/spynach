% Stage timer for comparison cases. Records the wall-clock time of named
% stages; the same labels are used on the spynach side, so the report can
% put the two next to each other. Syntax:
%
%     timer=cmp_timer();
%     timer.start('basis'); ...; timer.stop();
%     timer.records         % struct array with fields label, seconds
%
% Stages may not nest.

classdef cmp_timer < handle

    properties
        records=struct('label',{},'seconds',{});
    end

    properties (Access=private)
        label='';
        started=[];
    end

    methods

        function start(obj,label)
            if ~isempty(obj.started)
                error(['stage ''' obj.label ''' is still running, stages do not nest.']);
            end
            obj.label=label; obj.started=tic;
        end

        function seconds=stop(obj)
            if isempty(obj.started)
                error('no stage is running.');
            end
            seconds=toc(obj.started);
            obj.records(end+1)=struct('label',obj.label,'seconds',seconds);
            fprintf('%-32s %8.1f s\n',obj.label,seconds);
            obj.label=''; obj.started=[];
        end

    end

end
