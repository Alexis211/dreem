<TeXmacs|1.99.2>

<style|article>

<\body>
  <doc-data|<doc-title|Sleep Stage Classification, by Dreem>|<\doc-subtitle>
    Wavelets & Classification Course Project Report
  </doc-subtitle>|<doc-misc|<name|Alex Auvolat>>>

  <subsection|Task Presentation>

  In this challenge, provided by the start-up company Dreem, we have to
  identify the phases of sleep based on 15-second recordings of
  electroencephalogram (EEG) and accelerometer data. The data is recorded by
  a simple device the user puts on her head before going to sleep. The EEG is
  sampled at 250Hz, whereas the accelerometer is sampled at 10Hz. In
  Figure<nbsp><reference|typicaleeg> we show a typical EEG recording, and in
  Figure<nbsp><reference|typicalacc> we show a typical accelerometer
  recording, which contains three signals corredponding to the three axis of
  rotation of the user's head. The instances are to be categorized in one of
  the five sleep types described in Table<nbsp><reference|typetable>.

  <big-figure|<image|typical_eeg_REM.png|640px|480px||>|<label|typicaleeg>Typical
  EEG signal (REM sleep)>

  <big-table|<tabular|<tformat|<cwith|1|1|1|-1|cell-bborder|1px>|<cwith|6|6|1|-1|cell-bborder|1px>|<cwith|2|6|1|2|cell-halign|c>|<table|<row|<cell|<strong|Class>>|<cell|<strong|Code>>|<cell|<strong|Description>>|<cell|<strong|#train>>|<cell|<strong|#test>>>|<row|<cell|0>|<cell|>|<cell|Wake>|<cell|1342>|<cell|>>|<row|<cell|1>|<cell|N1>|<cell|Light
  sleep (``somnolence'')>|<cell|428>|<cell|>>|<row|<cell|2>|<cell|N2>|<cell|Intermediatesleep>|<cell|15334>|<cell|>>|<row|<cell|3>|<cell|N3>|<cell|Deep
  sleep>|<cell|9640>|<cell|>>|<row|<cell|4>|<cell|REM>|<cell|Paradoxical
  sleep (stage where dreams occur)>|<cell|4385>|<cell|>>|<row|<cell|>|<cell|>|<cell|Total>|<cell|31129>|<cell|30458>>>>>|<label|typetable>The
  classes in which the examples are to be classified.>

  \;

  \;

  <big-figure|<image|typical_acc_2_N2.png|640px|480px||>|<label|typicalacc>Typical
  accelerometer signal (N2 sleep), has three data channels for the three
  dimensions>

  <subsection|Feature Extraction>

  The signal is periodic, therefore a simple discrete Fourier transform seems
  to be a good choice for feature extraction. However, this method did not
  provide competitive results, so I added several more complexities to the
  feature extractor:

  <\itemize>
    <item>The Fourier coefficients are multiplied by the Fourier transform of
    a Ricker wavelet function, therefore effectively convolving the signal
    with the Ricker wavelet. This is done for a series of wavelet functions,
    effectively exacerbating some specific frequency ranges in the obtained
    coefficients.

    Les maths ICI

    <item>We calculate the square roots of the absolute values of these
    coefficients, in order to work on the energies of the different
    frequencies.

    <item>Since the coefficients are too many, and fluctuate very wildly, I
    apply a Gaussian filter on the coefficients. The variance of the Gaussian
    is adapted to the wavelet used to filter the data: for high-frequencies
    the variance is exponentially larger than for the low-frequencies, since
    there are exponentially more coefficients corresponding to a high
    frequency range than to a low frequency range.

    <item>The frequency domain is then subsampled. The Gaussian filter is
    usefull so as to make sure that the values of the coefficients that are
    dropped out are somehow integrated in the coefficients that are kept.

    <item>We then do a PCA on the coefficients, in order to find 3, 4 or 5
    principal components for each frequency band.

    <item>I also tried using a different PCA for the elements corresponding
    to different labels, which multiplies the number of coefficients in the
    feature vector by 5.

    <item>Overall, the feature vector contains
    <math|n<rsub|classes>*n<rsub|signals>*n<rsub|frequencies>n<rsub|components>>
    values, with typically:

    <\eqnarray*>
      <tformat|<table|<row|<cell|n<rsub|classes>>|<cell|=>|<cell|5>>|<row|<cell|n<rsub|signals>>|<cell|=>|<cell|4>>|<row|<cell|n<rsub|frequencies>>|<cell|\<simeq\>>|<cell|10>>|<row|<cell|n<rsub|components>>|<cell|\<simeq\>>|<cell|4>>>>
    </eqnarray*>

    Which makes a total of <math|800> features.
  </itemize>

  <subsection|Experimental Results>

  The features are then piped into a random forest which does the
  classification automatically using <math|100> trees. The results we obtain
  are presented in Table<nbsp><reference|results>.

  <big-table|<tabular|<tformat|<cwith|1|1|1|-1|cell-bborder|1px>|<cwith|2|-1|1|-1|cell-halign|r>|<table|<row|<cell|<strong|model>>|<cell|<strong|<math|n<rsub|freq><rsup|eeg>>>>|<cell|<strong|<math|n<rsub|comp><rsup|eeg>>>>|<cell|<strong|<math|n<rsub|freq><rsup|acc>>>>|<cell|<strong|<math|n<rsub|comp<rprime|''>><rsup|acc>>>>|<cell|<strong|Valid.
  Error Rate>>|<cell|<strong|Valid. Score>>|<cell|<strong|Test
  Score>>>|<row|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>>|<row|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>>|<row|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>>|<row|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>|<cell|>>>>>|<label|results>Results
  of various hyperparameter combinations>

  <subsection|Conclusion>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|1|1>>
    <associate|auto-3|<tuple|1|1>>
    <associate|auto-4|<tuple|2|2>>
    <associate|auto-5|<tuple|2|2>>
    <associate|auto-6|<tuple|3|2>>
    <associate|auto-7|<tuple|2|2>>
    <associate|auto-8|<tuple|4|?>>
    <associate|results|<tuple|2|?>>
    <associate|typetable|<tuple|1|?>>
    <associate|typicalacc|<tuple|2|?>>
    <associate|typicaleeg|<tuple|1|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|figure>
      <tuple|normal|Typical EEG signal (REM sleep)|<pageref|auto-2>>

      <tuple|normal|Typical accelerometer signal (N2 sleep), has three data
      channels for the three dimensions|<pageref|auto-4>>
    </associate>
    <\associate|table>
      <tuple|normal|The classes in which the examples are to be
      classified.|<pageref|auto-3>>
    </associate>
    <\associate|toc>
      <with|par-left|<quote|1tab>|1<space|2spc>Task Presentation
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1>>

      <with|par-left|<quote|1tab>|2<space|2spc>Feature Construction
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1tab>|3<space|2spc>Experimental Results
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1tab>|4<space|2spc>Conclusion
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>
    </associate>
  </collection>
</auxiliary>