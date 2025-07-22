<?php
require __DIR__.'/../vendor/autoload.php';

use Rindow\NeuralNetworks\Support\GenericUtils;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Layer\AbstractRNNLayer;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Data\Sequence\Tokenizer;
use Rindow\NeuralNetworks\Data\Sequence\Preprocessor;

# Download the file
class DateDataset
{
    protected $baseUrl = __DIR__."\\";
    protected $downloadFile = 'date.txt';

    public function __construct($mo,$inputTokenizer=null,$targetTokenizer=null)
    {
        $this->mo = $mo;
        $this->datasetDir = $this->getDatasetDir();
        if(!file_exists($this->datasetDir)) {
            @mkdir($this->datasetDir,0777,true);
        }
        $this->saveFile = $this->datasetDir . "/date.pkl";
        $this->preprocessor = new Preprocessor($mo);
    }

    protected function getDatasetDir()
    {
        return sys_get_temp_dir().'/rindow/nn/datasets/date';
    }

    protected function download($filename)
    {
        $filePath = $this->datasetDir . "/" . $filename;

        if(!file_exists($filePath)){
            $this->console("Downloading " . $filename . " ... ");
            copy($this->baseUrl.$filename, $filePath);
            $this->console("Done\n");
        }

        return $filePath;
    }

    public function preprocessSentence($w)
    {
        $w = '<start> '.$w.' <end>';
        return $w;
    }

    public function createDataset($path, $numExamples)
    {
        $contents = file_get_contents($path);
        $fp = fopen($path,"r");
        if($fp===false) {
            throw new InvalidArgumentException('file not found: '.$path);
        }
        $enSentences = [];
        $numSentences = [];
        while($line=fgets($fp)) {
            if($numExamples!==null) {
                $numExamples--;
                if($numExamples<0)
                    break;
            }
            $line = trim($line);
            $blocks = explode("_",$line);
            $enSentences[] = $blocks[0];
            $numSentences[] = $blocks[1];
        }
        return [$numSentences,$enSentences];
    }

    public function fitOnTexts($tokenizer,$texts)
    {
        foreach ($texts as $line) {
            $text = str_split($line);
            foreach ($text as $char) {
                if(!array_key_exists($char,$tokenizer->char2num)) {
                    $tokenizer->char2num[$char] = count($tokenizer->num2char);
                    $tokenizer->num2char[] = $char;
                }
            }
        }
    }

    public function textsToSequences($tokenizer,$texts)
    {
        $size = count($texts);
        $maxlen = array_reduce($texts,function($max,$t){ return max(strlen($t),$max);},-1);
        $maxlen += 2;
        $sequences = $this->mo->zeros([$size,$maxlen],NDArray::int32);
        foreach ($texts as $li => $line) {
            $sequences[$li][0] = $tokenizer->char2num['<start>'];
            foreach (str_split($line) as $col => $char) {
                $sequences[$li][$col+1] = $tokenizer->char2num[$char];
            }
            $sequences[$li][strlen($line)+1] = $tokenizer->char2num['<end>'];
        }
        return $sequences;
    }

    public function numWords($tokenizer)
    {
        return count($tokenizer->char2num);
    }

    public function wordToIndex($tokenizer,$char)
    {
        return $tokenizer->char2num[$char];
    }

    public function indexToWord($tokenizer,$num)
    {
        return $tokenizer->num2char[$num];
    }

    public function sequencesToTexts($tokenizer,$sequences)
    {
        [$size,$length] = $sequences->shape();
        $texts = [];
        foreach ($sequences as $seq) {
            $text = '';
            foreach ($seq as $num) {
                $text .= $tokenizer->num2char[$num];
            }
            $texts[] = $text;
        }
        return $texts;
    }

    public function tokenize($texts,$numWords=null,$tokenizer=null)
    {
        if($tokenizer==null) {
            $tokenizer = new \stdClass();
            $tokenizer->num2char = ['<null>','<start>','<end>'];
            $tokenizer->char2num = array_flip($tokenizer->num2char);
            $tokenizer->numWords = $numWords;
        }
        $this->fitOnTexts($tokenizer,$texts);
        $sequences = $this->textsToSequences($tokenizer,$texts);
        return [$sequences, $tokenizer];
    }

    protected function console($message)
    {
        fwrite(STDERR,$message);
    }

    public function loadData(
        string $path=null, int $numExamples=null, int $numWords=null)
    {
        if($path==null) {
            $path = $this->download($this->downloadFile);
        }
        # creating cleaned input, output pairs
        [$targ_lang, $inp_lang] = $this->createDataset($path, $numExamples);

        [$input_tensor, $inp_lang_tokenizer] = $this->tokenize($inp_lang,$numWords);
        [$target_tensor, $targ_lang_tokenizer] = $this->tokenize($targ_lang,$numWords);
        $numInput = $input_tensor->shape()[0];
        $choice = $this->mo->random()->choice($numInput,$numInput,$replace=false);
        $input_tensor = $this->shuffle($input_tensor,$choice);
        $target_tensor = $this->shuffle($target_tensor,$choice);

        return [$input_tensor, $target_tensor, $inp_lang_tokenizer, $targ_lang_tokenizer];
    }

    public function shuffle(NDArray $tensor, NDArray $choice) : NDArray
    {
        $result = $this->mo->zerosLike($tensor);
        $size = $tensor->shape()[0];
        for($i=0;$i<$size;$i++) {
            $this->mo->la()->copy($tensor[$choice[$i]],$result[$i]);
        }
        return $result;
    }

    public function convert($lang, NDArray $tensor) : void
    {
        $size = $tensor->shape()[0];
        for($i=0;$t<$size;$t++) {
            $t = $tensor[$i];
            if($t!=0)
                echo sprintf("%d ----> %s\n", $t, $lang->index_word[$t]);
        }
    }
}

class Encoder extends AbstractModel
{
    public function __construct(
        $backend,
        $builder,
        int $vocabSize,
        int $wordVectSize,
        int $units,
        int $inputLength
        )
    {
        $this->backend = $backend;
        $this->vocabSize = $vocabSize;
        $this->wordVectSize = $wordVectSize;
        $this->units = $units;
        $this->embedding = $builder->layers()->Embedding(
            $vocabSize,$wordVectSize,
            ['input_length'=>$inputLength]
        );
        $this->rnn = $builder->layers()->GRU(
            $units,
            ['return_state'=>true,'return_sequences'=>true,
             'recurrent_initializer'=>'glorot_uniform']
        );
    }

    protected function call(
        object $inputs,
        bool $training,
        array $initial_state=null,
        array $options=null
        ) : array
    {
        $K = $this->backend;
        $wordVect = $this->embedding->forward($inputs,$training);
        [$outputs,$states] = $this->rnn->forward(
            $wordVect,$training,$initial_state);
        return [$outputs, $states];
    }
}

class Decoder extends AbstractModel
{
    protected $backend;
    protected $vocabSize;
    protected $wordVectSize;
    protected $units;
    protected $targetLength;
    protected $embedding;
    protected $rnn;
    protected $attention;
    protected $concat;
    protected $dense;
    protected $attentionScores;

    public function __construct(
        $backend,
        $builder,
        int $vocabSize,
        int $wordVectSize,
        int $units,
        int $inputLength,
        int $targetLength
        )
    {
        $this->backend = $backend;
        $this->vocabSize = $vocabSize;
        $this->wordVectSize = $wordVectSize;
        $this->units = $units;
        $this->inputLength = $inputLength;
        $this->targetLength = $targetLength;
        $this->embedding = $builder->layers()->Embedding(
            $vocabSize, $wordVectSize,
            ['input_length'=>$targetLength]
        );
        $this->rnn = $builder->layers()->GRU($units,
            ['return_state'=>true,'return_sequences'=>true,
             'recurrent_initializer'=>'glorot_uniform']
        );
        $this->attention = $builder->layers()->Attention();
        $this->concat = $builder->layers()->Concatenate();
        $this->dense = $builder->layers()->Dense($vocabSize);
    }

    protected function call(
        object $inputs,
        bool $training,
        array $initial_state=null,
        array $options=null
        ) : array
    {
        $K = $this->backend;
        $encOutputs=$options['enc_outputs'];

        $x = $this->embedding->forward($inputs,$training);
        [$rnnSequence,$states] = $this->rnn->forward(
            $x,$training,$initial_state);

        $contextVector = $this->attention->forward(
            [$rnnSequence,$encOutputs],$training,$options);
        if(is_array($contextVector)) {
            [$contextVector,$attentionScores] = $contextVector;
            $this->attentionScores = $attentionScores;
        }
        $outputs = $this->concat->forward([$contextVector, $rnnSequence],$training);

        $outputs = $this->dense->forward($outputs,$training);
        return [$outputs,$states];
    }

    public function getAttentionScores()
    {
        return $this->attentionScores;
    }
}


class Seq2seq extends AbstractModel
{
    public function __construct(
        $mo,
        $backend,
        $builder,
        $inputLength=null,
        $inputVocabSize=null,
        $outputLength=null,
        $targetVocabSize=null,
        $wordVectSize=8,
        $units=256,
        $startVocId=0,
        $endVocId=0,
        $plt=null
        )
    {
        parent::__construct($backend,$builder);
        $this->encoder = new Encoder(
            $backend,
            $builder,
            $inputVocabSize,
            $wordVectSize,
            $units,
            $inputLength
        );
        $this->decoder = new Decoder(
            $backend,
            $builder,
            $targetVocabSize,
            $wordVectSize,
            $units,
            $inputLength,
            $outputLength
        );
        $this->out = $builder->layers()->Activation('softmax');
        $this->mo = $mo;
        $this->backend = $backend;
        $this->startVocId = $startVocId;
        $this->endVocId = $endVocId;
        $this->inputLength = $inputLength;
        $this->outputLength = $outputLength;
        $this->units = $units;
        $this->plt = $plt;
    }

    protected function call($inputs, $training, $trues)
    {
        $K = $this->backend;
        [$encOutputs,$states] = $this->encoder->forward($inputs,$training);
        $options = ['enc_outputs'=>$encOutputs];
        [$outputs,$dmyStatus] = $this->decoder->forward($trues,$training,$states,$options);
        $outputs = $this->out->forward($outputs,$training);
        return $outputs;
    }

    public function shiftLeftSentence(
        NDArray $sentence
        ) : NDArray
    {
        $K = $this->backend;
        $shape = $sentence->shape();
        $batchs = $shape[0];
        $zeroPad = $K->zeros([$batchs,1],$sentence->dtype());
        $seq = $K->slice($sentence,[0,1],[-1,-1]);
        $result = $K->concat([$seq,$zeroPad],$axis=1);
        return $result;
    }

    protected function trueValuesFilter(NDArray $trues) : NDArray
    {
        return $this->shiftLeftSentence($trues);
    }

    public function predict(NDArray $inputs, array $options=null) : NDArray
    {
        $K = $this->backend;
        $attentionPlot = $options['attention_plot'];
        //$tmpAttentionPlot = $this->mo->zerosLike($attentionPlot);
        $inputs = $K->array($inputs);

        if($inputs->ndim()!=2) {
            throw new InvalidArgumentException('inputs shape must be 2D.');
        }
        $batchs = $inputs->shape()[0];
        if($batchs!=1) {
            throw new InvalidArgumentException('num of batch must be one.');
        }
        $status = [$K->zeros([$batchs, $this->units])];
        [$encOutputs, $status] = $this->encoder->forward($inputs, $training=false, $status);

        $decInputs = $K->array([[$this->startVocId]],$inputs->dtype());

        $result = [];
        $this->setShapeInspection(false);
        for($t=0;$t<$this->outputLength;$t++) {
            [$predictions, $status] = $this->decoder->forward(
                $decInputs, $training=false, $status,
                ['enc_outputs'=>$encOutputs,'return_attention_scores'=>true]);

            # storing the attention weights to plot later on
            $scores = $this->decoder->getAttentionScores();
            $this->mo->la()->copy(
                $K->ndarray($scores->reshape([$this->inputLength])),
                $attentionPlot[$t]);

            $predictedId = $K->scalar($K->argmax($predictions[0][0]));

            $result[] = $predictedId;

            if($this->endVocId == $predictedId) {
                $t++;
                break;
            }

            # the predicted ID is fed back into the model
            $decInputs = $K->array([[$predictedId]],$inputs->dtype());
        }
        //for($tt=0;$tt<$t;$tt++) {
        //    $this->mo->la()->copy($tmpAttentionPlot[$t-$tt-1],$attentionPlot[$tt]);
        //}

        $this->setShapeInspection(true);
        $result = $K->array([$result],NDArray::int32);
        #return result, sentence, attention_plot
        return $K->ndarray($result);
    }

    public function plotAttention(
        $attention, $sentence, $predictedSentence)
    {
        $plt = $this->plt;
        $config = [
            'frame.xTickPosition'=>'up',
            'frame.xTickLabelAngle'=>90,
            'figure.topMargin'=>100,
        ];
        $plt->figure(null,null,$config);
        #attention = attention[:len(predicted_sentence), :len(sentence)]
        $sentenceLen = count($sentence);
        $predictLen = count($predictedSentence);
        $image = $this->mo->zeros([$predictLen,$sentenceLen],$attention->dtype());
        for($y=0;$y<$predictLen;$y++) {
            for($x=0;$x<$sentenceLen;$x++) {
                $image[$y][$x] = $attention[$y][$x];
            }
        }
        $plt->imshow($image, $cmap='viridis',null,null,$origin='upper');

        //$sentence = array_pad($sentence,$this->inputLength, '');
        //$sentence = array_reverse($sentence);
        $plt->xticks($this->mo->arange(count($sentence)),$sentence);
        //$predictedSentence = array_pad($predictedSentence,$this->outputLength, '');
        $predictedSentence = array_reverse($predictedSentence);
        $plt->yticks($this->mo->arange(count($predictedSentence)),$predictedSentence);
    }
}

$numExamples=20000;#30000
$numWords=null;
$epochs = 10;
$batchSize = 64;
$wordVectSize=64;#256
$units=256;#1024


$mo = new MatrixOperator();
$nn = new NeuralNetworks($mo);
$pltConfig = [
];
$plt = new Plot($pltConfig,$mo);

$dataset = new DateDataset($mo);

echo "Generating data...\n";
[$inputTensor, $targetTensor, $inpLang, $targLang]
    = $dataset->loadData(null,$numExamples,$numWords);
$valSize = intval(floor(count($inputTensor)/10));
$trainSize = count($inputTensor)-$valSize;
$inputTensorTrain  = $inputTensor[[0,$trainSize-1]];
$targetTensorTrain = $targetTensor[[0,$trainSize-1]];
$inputTensorVal  = $inputTensor[[$trainSize,$valSize+$trainSize-1]];
$targetTensorVal = $targetTensor[[$trainSize,$valSize+$trainSize-1]];

$inputLength  = $inputTensor->shape()[1];
$outputLength = $targetTensor->shape()[1];
$inputVocabSize = $dataset->numWords($inpLang);
$targetVocabSize = $dataset->numWords($targLang);
$corpusSize = count($inputTensor);

echo "num_examples: $numExamples\n";
echo "num_words: $numWords\n";
echo "epoch: $epochs\n";
echo "batchSize: $batchSize\n";
echo "embedding_dim: $wordVectSize\n";
echo "units: $units\n";
echo "Total questions: $corpusSize\n";
echo "Input  word dictionary: $inputVocabSize(".$dataset->numWords($inpLang,true).")\n";
echo "Target word dictionary: $targetVocabSize(".$dataset->numWords($targLang,true).")\n";
echo "Input length: $inputLength\n";
echo "Output length: $outputLength\n";

echo "Sample Input:".$dataset->sequencesToTexts($inpLang,$inputTensor[[0,0]])."\n";
echo "Sample Output:".$dataset->sequencesToTexts($targLang,$targetTensor[[0,0]])."\n";


$seq2seq = new Seq2seq(
    $mo,
    $nn->backend(),
    $nn,
    $inputLength,
    $inputVocabSize,
    $outputLength,
    $targetVocabSize,
    $wordVectSize,
    $units,
    $dataset->wordToIndex($targLang,'<start>'),
    $dataset->wordToIndex($targLang,'<end>'),
    $plt
);

echo "Compile model...\n";
$seq2seq->compile([
    'loss'=>'sparse_categorical_crossentropy',
    'optimizer'=>'adam',
    'metrics'=>['accuracy','loss'],
]);
$seq2seq->summary();

$modelFilePath = __DIR__."/date-translation-with-attention.model";

if(file_exists($modelFilePath)) {
    echo "Loading model...\n";
    $seq2seq->loadWeightsFromFile($modelFilePath);
} else {
    echo "Train model...\n";
    $history = $seq2seq->fit(
        $inputTensorTrain,
        $targetTensorTrain,
        [
            'batch_size'=>$batchSize,
            'epochs'=>$epochs,
            'validation_data'=>[$inputTensorVal,$targetTensorVal],
            #callbacks=[checkpoint],
        ]);
    $seq2seq->saveWeightsToFile($modelFilePath);

    $plt->figure();
    $plt->plot($mo->array($history['accuracy']),null,null,'accuracy');
    $plt->plot($mo->array($history['val_accuracy']),null,null,'val_accuracy');
    $plt->plot($mo->array($history['loss']),null,null,'loss');
    $plt->plot($mo->array($history['val_loss']),null,null,'val_loss');
    $plt->legend();
    $plt->title('date-attention-translation');
}

$choice = $mo->random()->choice($corpusSize,10,false);
foreach($choice as $idx)
{
    $question = $inputTensor[$idx]->reshape([1,$inputLength]);
    $attentionPlot = $mo->zeros([$outputLength, $inputLength]);
    $predict = $seq2seq->predict(
        $question,['attention_plot'=>$attentionPlot]);
    $answer = $targetTensor[$idx]->reshape([1,$outputLength]);;
    $sentence = $dataset->sequencesToTexts($inpLang,$question)[0];
    $predictedSentence = $dataset->sequencesToTexts($targLang,$predict)[0];
    $targetSentence = $dataset->sequencesToTexts($targLang,$answer)[0];
    echo "Input:   $sentence\n";
    echo "Predict: $predictedSentence\n";
    echo "Target:  $targetSentence\n";
    echo "\n";
    #attention_plot = attention_plot[:len(predicted_sentence.split(' ')), :len(sentence.split(' '))]
    $q = array_map(
        function($n) use ($dataset,$inpLang) {
            return $dataset->indexToWord($inpLang,$n);
        },$question[0]->toArray());
    $p = array_map(
        function($n) use ($dataset,$targLang) {
            return $dataset->indexToWord($targLang,$n);
        },$predict[0]->toArray());
    $seq2seq->plotAttention($attentionPlot, $q, $p);
}
$plt->show();
