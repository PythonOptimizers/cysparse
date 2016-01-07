
..  _cysparse_compared_to_other_sparse_libraries:

==================================================================
:program:`CySparse` compared to other sparse matrix libraries
==================================================================

..	only:: html

	We compare briefly :program:`CySparse` with other existing libraries. If you are a user of one or several of these libraries, this page can also help you do the switch to :program:`CySparse` as we compare how things are done
	both in :program:`CySparse` and your preferred library.

..	only:: latex

	We compare briefly :program:`CySparse` with other existing libraries. If you are a user of one or several of these libraries, this chapter can also help you do the switch to :program:`CySparse` as we compare how things are done
	both in :program:`CySparse` and your preferred library.

:program:`NumPy`
================

:program:`NumPy` does **not** deal with sparse matrices, only dense ones but as it probably is **the** matrix library in :program:`Python` it is worth mentionning some similarities and discrepancies between the two. 

:program:`SciPy` sparse matrices aka :program:`scipy.sparses`
==============================================================

:program:`SciPy` can been seen as the little scientific brother of :program:`NumPy` [#scipy_defined_by_itself]_: :program:`NumPy` provides dense matrices while :program:`SciPy` adds scientific routines on top of it.
Both libraries, :program:`NumPy` and :program:`SciPy` are deeply intermangled and sometimes overlapping. As :program:`NumPy`, :program:`SciPy` is huge [#scipy_huge_vs_numpy]_ and we dont' provide as many fonctionnalities.
It is not our goal either. So let's compare what can be compared.

Performance
------------

:program:`PySparse`
====================

Performance
------------




..	only:: html
	
	..	rubric:: Footnotes


..	[#scipy_defined_by_itself] Although on `www.scipy.org <http://www.scipy.org/>`_, :program:`SciPy` is presented as *SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering* 
	and :program:`NumPy` is described as one of its main core packages, i.e. :program:`NumPy` is a sub library of :program:`SciPy`.

..	[#scipy_huge_vs_numpy] Of course, because :program:`SciPy` contains :program:`NumPy`!

