import scala.io.Source
import scala.collection.mutable
import scala.util.Random
import java.io._
import scala.util.Try

// i didn't realize i needed to call it here and build it within task3 itself lol 
// reference blackbox code provided by the assignment
// copy of blackbox.scala
case class Blackbox() {
  private val r1 = scala.util.Random
  r1.setSeed(553)

  def ask(filename: String, num: Int): Array[String] = {
    val input_file_path = filename

    val lines = Source.fromFile(input_file_path).getLines().toArray
    val stream = new Array[String](num)
    for (i <- 0 to num - 1) {
      stream(i) = lines(r1.nextInt(lines.length))
    }
    stream
  }
}


object task3 {

  // =========================
  // global adjustables/variables
  // =========================

  //100 provided by assignment 
  val reservoir_size: Int = 100

  //making these global because it needs to be maintained over iterations of calling reservoir sampling
  val reservoir: mutable.ArrayBuffer[String] = mutable.ArrayBuffer[String]()
  var sequence_num: Long = 0L

  // call random once,
  // trying to set immediately post creation now 
  val random_object: Random = Random

  // =========================
  // typical generic easy utilities
  // =========================

  //old checkinputs, modified for this assignment
  def check_inputs(args: Array[String]): (String, Int, Int, String) = {
    if (args.length != 4) {
      System.err.println("Usage: task3.py <input_filename> stream_size num_of_asks <output_filename>")
      sys.exit(1)
    }

    //filter threshold, input file name, output file name
    val input_filename = args(0)
    val stream_size = Try(args(1).toInt).getOrElse {
      System.err.println("stream_size must be an integer")
      sys.exit(1); 0
    }
    val num_of_asks = Try(args(2).toInt).getOrElse {
      System.err.println("num_of_asks must be an integer")
      sys.exit(1); 0
    }
    val output_filename = args(3)

    (input_filename, stream_size, num_of_asks, output_filename)
  }

  //also repeat write function 
  def write_output(output_filename: String, rows: Seq[Seq[String]]): Unit = {
    val writer = new PrintWriter(new File(output_filename))
    try {
      // copied column names in the sample
      writer.println("seqnum,0_id,20_id,40_id,60_id,80_id")
      rows.foreach { row =>
        writer.println(row.mkString(","))
      }
    } finally {
      writer.close()
    }
  }

  // =========================
  // reservoir sampling functions
  // =========================

  //process uids for reservoir sampling
  def process_users(users: Array[String]): (Long, Seq[String]) = {

    //iterate through each user in current stream (users from blackbox)
    users.foreach { uid =>
      //inc seq each iteration 
      sequence_num += 1

      //fill reservoir initially if smaller than reservoir 
      if (sequence_num <= reservoir_size) {

        //when not > 100, just use all 100
        reservoir.append(uid)
      } else {
        //sampling after first 100 of users
        //for the nth count user we get, keep with probability 100/n
        val keep_prob = reservoir_size.toFloat / sequence_num.toFloat

        //use rng.nextFloat similar to python random.random
        //found online, maybe will fix on vocareum? 
        // changed naming to random_object too to match naming 
        if (random_object.nextFloat() < keep_prob) {
          val index = random_object.nextInt(reservoir_size)
          reservoir(index) = uid
        }
      }
    }

    // after processing the batch, collect position that they are on/at
    // by the time we finish the first batch (stream_size >= 100),
    // reservoir will be full, so these indices are valid.
    val indices = Array(0, 20, 40, 60, 80)

    //should be the users/ids at 1, 21, 41, 61, 81 out of 100 
    val selected_ids = indices.map(i => reservoir(i))

    (sequence_num, selected_ids.toSeq)
  }

  // =========================
  // main sawce
  // =========================

  def main(args: Array[String]): Unit = {
    val (input_filename, stream_size, num_of_asks, output_filename) = check_inputs(args)

    val bx = Blackbox()

    //store results for output
    val results = mutable.ArrayBuffer[Seq[String]]()

    //iterate through each ask and build to write 
    for (_ <- 0 until num_of_asks) {
      val users = bx.ask(input_filename, stream_size)
      val (seq_after_batch, selected_ids) = process_users(users)
      val row: Seq[String] = Seq(seq_after_batch.toString) ++ selected_ids
      results.append(row)
    }

    write_output(output_filename, results.toSeq)
  }
}