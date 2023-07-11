<TeXmacs|2.1>

<style|generic>

<\body>
  <doc-data|<doc-title|ECMI 2023 - Hydro>>

  <section|Choosing training data>

  \;

  Our first idea is to try and systematically eliminate gauge stations which
  do not have an effect on the water level of szeged within 7 days.

  Correlation Method: Pearson?

  <\verbatim>
    filtered = {}

    for every station s:

    <space|1em>for d in 0 to ~14:

    \ \ \ \ calculate correlation between water level in szeged and station
    s.

    \ \ calculate time t it takes for the water at s to reach szeged.

    \ \ if t \<less\> t_max:

    \ \ \ \ add s to filtered
  </verbatim>

  \;

  Check the filtered stations on the map.
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1|../../../../../.TeXmacs/texts/scratch/no_name_4.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Choosing
      training data> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>