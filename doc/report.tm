<TeXmacs|1.99.2>

<style|article>

<\body>
  \;

  \;

  \;

  <center|<\name>
    Sparse Wavelet Representations Class

    Challenge Data

    \;

    Master 2 Mathématiques, Vision & Apprentissage

    École Normale Supérieure de Cachan
  </name>>

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  <doc-data|<doc-title|Sleep Stage Classification using Wavelet
  Transforms>|<\doc-subtitle>
    Project Report

    \;
  </doc-subtitle>|<\doc-date>
    \;

    \;

    \;

    Febuary 2016

    \;

    \;

    \;

    <name|Alex Auvolat>

    <verbatim|alex.auvolat@ens.fr>

    <name|Département d'Informatique>

    <name|École Normale Supérieure>

    \;

    \;

    \;

    \;

    \;

    \;

    \;

    \;
  </doc-date>|<\doc-misc>
    <em|Data & challenge provided by>

    \;

    <\name>
      Dreem
    </name>

    <verbatim|http://www.dreem.com/>
  </doc-misc>>

  \;

  \;

  \;

  \;

  <new-page>

  <section|Task Presentation>

  <paragraph|Introduction.>In this challenge, provided by the start-up
  company Dreem, we have to identify the phases of sleep based on 15-second
  recordings of electroencephalogram (EEG) and accelerometer data. The data
  is recorded by a simple device the user puts on her head before going to
  sleep. In Figure<nbsp><reference|typicaleeg> we show a typical EEG
  recording, and in Figure<nbsp><reference|typicalacc> we show a typical
  accelerometer recording, which contains three signals corredponding to the
  three axis of rotation of the user's head.

  <paragraph|Data technicalities.>Each data sample is composed of one EEG
  signal and of three accelerometer signal. These signals are described
  precisely in Table<nbsp><reference|tblsig>. \ The training examples belong
  to 5 classes which are described in Table<nbsp><reference|typetable>.

  <big-figure|<image|typical_eeg_REM.png|640px|480px||>|<label|typicaleeg>Typical
  EEG signal (REM sleep)>

  <paragraph|Difficulties.>Several difficulties are found in the dataset:

  <\itemize>
    <item>The signals are quite noisy, due to the use of a cheap recorder
    that doesn't provide us with optimal signal quality.

    <item>The number of classes is not very well balanced throughout the
    dataset.
  </itemize>

  <big-table|<tabular|<tformat|<cwith|2|-1|2|-1|cell-halign|r>|<cwith|1|1|1|-1|cell-bborder|1px>|<cwith|3|3|1|-1|cell-bborder|1px>|<table|<row|<cell|<strong|Signal>>|<cell|<strong|Channels>>|<cell|<strong|Sampling
  freq.>>|<cell|<strong|Number of points>>>|<row|<cell|Electroencephalogram>|<cell|1>|<cell|250
  Hz>|<cell|3750>>|<row|<cell|Accelerometer>|<cell|3>|<cell|10
  Hz>|<cell|350>>|<row|<cell|<strong|Total>>|<cell|>|<cell|>|<cell|4100>>>>>|<label|tblsig>Technical
  details of the signal.>

  \;

  <big-table|<tabular|<tformat|<cwith|1|1|1|-1|cell-bborder|1px>|<cwith|6|6|1|-1|cell-bborder|1px>|<cwith|2|6|1|2|cell-halign|c>|<table|<row|<cell|<strong|Class>>|<cell|<strong|Code>>|<cell|<strong|Description>>|<cell|<strong|#train>>|<cell|<strong|#test>>>|<row|<cell|0>|<cell|>|<cell|Wake>|<cell|1342>|<cell|>>|<row|<cell|1>|<cell|N1>|<cell|Light
  sleep (``somnolence'')>|<cell|428>|<cell|>>|<row|<cell|2>|<cell|N2>|<cell|Intermediate
  sleep>|<cell|15334>|<cell|>>|<row|<cell|3>|<cell|N3>|<cell|Deep
  sleep>|<cell|9640>|<cell|>>|<row|<cell|4>|<cell|REM>|<cell|Paradoxical
  sleep (stage where dreams occur)>|<cell|4385>|<cell|>>|<row|<cell|>|<cell|>|<cell|Total>|<cell|31129>|<cell|30458>>>>>|<label|typetable>The
  classes in which the examples are to be classified.>

  <big-figure|<image|typical_acc_2_N2.png|640px|480px||>|<label|typicalacc>Typical
  accelerometer signal (N2 sleep), has three data channels for the three
  dimensions>

  <section|Feature Extraction>

  <paragraph|Properties of the signal.>The signal is periodic, therefore
  solutions based on the discrete Fourier transform seem to be a good general
  choice. Ignoring the argument of the Fourier coefficients (which correspond
  to the phase of the signal) and taking only their modulus creates
  invariance in the representation and enables the model to generalize.

  <paragraph|Several wavelet transforms for feature extraction.>The first
  method was simply to do an FFT on the signal and to try to classify the
  signals based on that, but this method did not give competitive results. To
  try and get better results than the simple FFT approach, I improved the
  feature extractor to use Wavelet transforms.

  The signal is first convolved with a Ricker wavelet. This is done for a
  series of wavelet functions at different frequencies. The Ricker wavelets I
  use are showed in Figure<nbsp><reference|rickerwv>.

  The equation of a Ricker wavelet is:

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<psi\><rsub|\<sigma\>><around*|(|t|)>>|<cell|=>|<cell|<frac|2|<sqrt|3\<sigma\>>\<pi\><rsup|1/4>><around*|(|1-<frac|t<rsup|2>|\<sigma\><rsup|2>>|)>e<rsup|<frac|-t<rsup|2>|2\<sigma\><rsup|2>>>>>>>
  </eqnarray*>

  The frequency bands are typically spaced by
  <math|\<sigma\><rsub|i+1>/\<sigma\><rsub|i>=2>, but I also used
  <math|\<sigma\><rsub|i+1>/\<sigma\><rsub|i>=<sqrt|2>> in some of my
  experiments.

  <paragraph|Remarks on the Wavelet transform.>A convolution by a Wavelet
  transform in the time domain is equivalent to a multiplication in the
  Fourier domain, which has several implications:

  <\itemize>
    <item>The convolution can be done quickly, in <math|O<around*|(|n*log
    n|)>> (time required to do a FFT), instead of
    <math|O<around*|(|n<rsup|2>|)>> for a naive approach. Even better: if we
    are only interested in the Fourier coefficients of the convonleved
    signal, then we can do <math|k> wavelet transforms in time
    <math|O<around*|(|n*log n+n*k|)>> instead of <math|O<around*|(|n*k*log
    n|)>> by doing the FFT only once for all the convolutions.

    <item>The Wavelet transform can be interpreted as effectively
    exacerbating some specific frequency ranges in the Fourier domain. The
    Fourier transform of the wavelets that I used are showed in
    Figure<nbsp><reference|rickerfft>, and we observe clearly that the
    wavelet span different frequency ranges.
  </itemize>

  <paragraph|Next steps of data processing.>It is impossible to run a
  classifier on <math|3750\<times\>10> coefficients, which is the number of
  coefficients we obtain if we convolve the signal with 10 different Wavelet
  functions. In order to use the data efficiently, we must reduce the
  dimensionnality of the signal:

  <\itemize>
    <item>I calculate the absolute value (modulus) of the coefficients, and
    take their square root to bring them all in similar amplitud ranges.

    <item>We then do a PCA on the coefficients, in order to find 5 to 15
    principal components for each transformed signal.

    <item>Overall, the feature vector contains
    <math|n<rsub|signals>*n<rsub|frequencies>n<rsub|components>> values, with
    typically:

    <\eqnarray*>
      <tformat|<table|<row|<cell|n<rsub|signals>>|<cell|=>|<cell|4>>|<row|<cell|n<rsub|frequencies>>|<cell|\<simeq\>>|<cell|10>>|<row|<cell|n<rsub|components>>|<cell|\<simeq\>>|<cell|4>>>>
    </eqnarray*>

    Which makes a total of <math|100> to <math|1000> features.
  </itemize>

  <big-figure|<image|half_wavelets.png|640px|480px||>|<label|rickerwv>Ricker
  wavelets used in one of the models (50% lowest frequencies).>

  <big-figure|<image|half_wavelets_fft.png|640px|480px||>|<label|rickerfft>Fourier
  transform of the Ricker wavelets of the same model (50% highest
  frequencies).>

  <section|Experimental Results>

  <paragraph|Random forest classification.>The features are then fed into a
  random forest classifier which does the classification automatically using
  <math|100> trees. The results we obtain are presented in
  Table<nbsp><reference|results>.

  <big-table|<tabular|<tformat|<cwith|1|1|1|-1|cell-bborder|1px>|<cwith|2|-1|1|-1|cell-halign|r>|<table|<row|<cell|<strong|model>>|<cell|<strong|<math|n<rsub|freq><rsup|eeg>>>>|<cell|<strong|<math|n<rsub|PC><rsup|eeg>>>>|<cell|<strong|<math|n<rsub|freq><rsup|acc>>>>|<cell|<strong|<math|n<rsub|PC><rsup|acc>>>>|<cell|<strong|Valid.
  Error Rate>>|<cell|<strong|Valid. Score>>|<cell|<strong|Test
  Score>>>|<row|<cell|7>|<cell|22>|<cell|15>|<cell|12>|<cell|10>|<cell|0.1767>|<cell|>|<cell|<strong|0.8643>>>|<row|<cell|22>|<cell|11>|<cell|5>|<cell|14>|<cell|5>|<cell|0.1602>|<cell|>|<cell|0.8573>>|<row|<cell|36>|<cell|6>|<cell|3>|<cell|8>|<cell|4>|<cell|<strong|0.1477>>|<cell|0.8813>|<cell|0.8557>>>>>|<label|results>Results
  of various hyperparameter combinations>

  <paragraph|Hyperparameter search.>I tried several parameters for the
  models, in particular I varied the selection of Wavelet functions with
  frequency bands overlapping more or less. I tried a large variety of
  parameters on a local validation set of 6129 examples that were removed
  from the training set. Although I only submitted 4 solutions to the public
  leaderboard, I found that it was very easy to overfit the validation set
  with too much tuning on the model. The best public results were in fact
  obtained with one of the first models I tried.

  <paragraph|Best solution.>The best model uses a large number of different
  wavelet: 22 for the EEG, spaced by <math|\<sigma\><rsub|i+1>/\<sigma\><rsub|i>=<sqrt|2>>
  with <math|\<sigma\><rsub|1>=1> and <math|\<sigma\><rsub|22>=1448>, and 12
  for the accelerometer spaced by <math|\<sigma\><rsub|i+1>/\<sigma\><rsub|i>=2>
  with <math|\<sigma\><rsub|1>=1> and <math|\<sigma\><rsub|11>=1024>. The
  feature vectors obtained for the data on a few training examples are shown
  in Figure<nbsp><reference|figfeatures>.

  <paragraph|Number of trees.>The quality of the solution provided by a
  random forest can be improved slightly by using more random trees. However
  above a certain point the computationnal cost becomes prohibitive and the
  gains are minimal. For my experiments, I settled with a compromise of 128
  random trees.

  <big-figure|<image|model50/__features.png|640px|480px||>|<label|figfeatures>Features
  obtained on a few of the training samples.>

  <paragraph|Linear classifier vs. random forest classifier.>I also tried to
  use a linear classifier (one vs. all Logistic regression), which didn't
  work as well as the random forest classifier: the error rate was of at
  least 22% on the validation set, whereas the validation error with a random
  forest classifier was of about 15%.

  <paragraph|Concluding remarks.>

  <\itemize>
    <item>We observe that a simple Wavelet transform is usefull to separate
    the frequency ranges in the signal, and enables us to classify the
    signals quite well.

    <item>With this simple method I ranked 3<rsup|rd> on the leaderboard,
    which is not a very impressive feat given the small number of
    participants.

    <item>The linear classifier experiment however shows that this feature
    extractor is not sufficient to make the problem linearly separable. More
    powerfull feature extractors remain to be designed.
  </itemize>
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|2>>
    <associate|auto-10|<tuple|4|3>>
    <associate|auto-11|<tuple|5|3>>
    <associate|auto-12|<tuple|6|4>>
    <associate|auto-13|<tuple|7|4>>
    <associate|auto-14|<tuple|3|4>>
    <associate|auto-15|<tuple|4|5>>
    <associate|auto-16|<tuple|3|5>>
    <associate|auto-17|<tuple|8|5>>
    <associate|auto-18|<tuple|3|5>>
    <associate|auto-19|<tuple|9|5>>
    <associate|auto-2|<tuple|1|2>>
    <associate|auto-20|<tuple|10|5>>
    <associate|auto-21|<tuple|11|5>>
    <associate|auto-22|<tuple|5|5>>
    <associate|auto-23|<tuple|12|6>>
    <associate|auto-24|<tuple|13|6>>
    <associate|auto-25|<tuple|7|?>>
    <associate|auto-3|<tuple|2|2>>
    <associate|auto-4|<tuple|1|2>>
    <associate|auto-5|<tuple|3|2>>
    <associate|auto-6|<tuple|1|2>>
    <associate|auto-7|<tuple|2|3>>
    <associate|auto-8|<tuple|2|3>>
    <associate|auto-9|<tuple|2|3>>
    <associate|figfeatures|<tuple|5|6>>
    <associate|results|<tuple|3|5>>
    <associate|rickerfft|<tuple|4|5>>
    <associate|rickerwv|<tuple|3|4>>
    <associate|tblsig|<tuple|1|2>>
    <associate|typetable|<tuple|2|3>>
    <associate|typicalacc|<tuple|2|3>>
    <associate|typicaleeg|<tuple|1|2>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|figure>
      <tuple|normal|Typical EEG signal (REM sleep)|<pageref|auto-4>>

      <tuple|normal|Typical accelerometer signal (N2 sleep), has three data
      channels for the three dimensions|<pageref|auto-8>>

      <tuple|normal|Ricker wavelets used in one of the models (50% lowest
      frequencies).|<pageref|auto-14>>

      <tuple|normal|Fourier transform of the Ricker wavelets of the same
      model (50% highest frequencies).|<pageref|auto-15>>

      <tuple|normal||<pageref|auto-16>>

      <tuple|normal||<pageref|auto-17>>

      <tuple|normal|Features obtained on a few of the training
      samples.|<pageref|auto-23>>
    </associate>
    <\associate|table>
      <tuple|normal|Technical details of the signal.|<pageref|auto-6>>

      <tuple|normal|The classes in which the examples are to be
      classified.|<pageref|auto-7>>

      <tuple|normal|Results of various hyperparameter
      combinations|<pageref|auto-20>>
    </associate>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Task
      Presentation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|4tab>|Introduction.
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.15fn>>

      <with|par-left|<quote|4tab>|Data technicalities.
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.15fn>>

      <with|par-left|<quote|4tab>|Difficulties.
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.15fn>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Feature
      Extraction> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9><vspace|0.5fn>

      <with|par-left|<quote|4tab>|Properties of the signal.
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10><vspace|0.15fn>>

      <with|par-left|<quote|4tab>|Several wavelet transforms for feature
      extraction. <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11><vspace|0.15fn>>

      <with|par-left|<quote|4tab>|Remarks on the Wavelet transform.
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-12><vspace|0.15fn>>

      <with|par-left|<quote|4tab>|Next steps of data processing.
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-13><vspace|0.15fn>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Experimental
      Results> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-18><vspace|0.5fn>

      <with|par-left|<quote|4tab>|Random forest classification.
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-19><vspace|0.15fn>>

      <with|par-left|<quote|4tab>|Hyperparameter search.
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-21><vspace|0.15fn>>

      <with|par-left|<quote|4tab>|Best solution.
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-22><vspace|0.15fn>>

      <with|par-left|<quote|4tab>|Linear classifier vs. random forest
      classifier. <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-24><vspace|0.15fn>>
    </associate>
  </collection>
</auxiliary>