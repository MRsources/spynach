% Shadow of Spinach kernel/utilities/isworkernode.m for fixture gene-
% ration. Behaves as the original, except that it returns true when
% called directly from evolution(). This sends evolution() into its
% serial worker-node branch instead of the Kuprov-Edwards split, which
% needs distributed arrays; MATLAB's Parallel Computing Toolbox does
% not support these on local process pools on Apple silicon Macs.
% The two branches compute the same trajectory with the same propa-
% gator() call. All other callers (create, basis, operator, hamilto-
% nian, propagator) see the unmodified value, so the parallel pool
% behaves as in stock Spinach. Spinach's startup check existentials()
% rejects any shadow of a kernel function, so dump scripts that rely
% on this file set sys.disable={'hygiene'}; that switch also skips
% cache tidying and GPU resets, neither of which affects results.
% Syntax:
%
%                    answer=isworkernode()
%
% Parameters:
%
%    none
%
% Outputs:
%
%    answer - true if running on a parallel worker process, or if
%             the immediate caller is evolution()

function answer=isworkernode()

    % Original Spinach behaviour
    answer=parallel.internal.pool.isPoolWorker();

    % Serial branch for evolution() on the head node
    if ~answer
        call_stack=dbstack(1);
        answer=(~isempty(call_stack))&&strcmp(call_stack(1).name,'evolution');
    end

end
