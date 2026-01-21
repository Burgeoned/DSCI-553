import scala.collection.mutable
import java.io._
import scala.util.Try

object task2 {

  //a lot can be reused from task1

  // =========================
  // global adjustables/variables
  // =========================

  //large prime for hashing
  val prime_number: Long = 2147483647L

  //hash funcs for the flajolet martin algo
  val num_hash_functions: Int = 100

  //hash func must be divisible by group size
  val group_size: Int = 10
  val num_groups: Int = num_hash_functions / group_size

  //odd numbers    
  val hash_params: Array[(Int, Int)] =
    Array.tabulate(num_hash_functions)(i => (2 * i + 1, 2 * i + 3))

  // =========================
  // typical generic easy utilities
  // =========================

  //old checkinputs, modified for this assignment
  def check_inputs(args: Array[String]): (String, Int, Int, String) = {
    if (args.length != 4) {
      System.err.println("Usage: task2.py <input_filename> stream_size num_of_asks <output_filename>")
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

  //outputs are time, ground truth, estimation
  def write_output(output_filename: String, rows: Seq[(Int, Int, Int)]): Unit = {
    val writer = new PrintWriter(new File(output_filename))
    try {
      writer.println("Time,Ground Truth,Estimation")
      rows.foreach { case (time, ground_truth, estimation) =>
        writer.println(s"$time,$ground_truth,$estimation")
      }
    } finally {
      writer.close()
    }
  }

  // =========================
  // math/ flajolet martin algorithm functions
  // =========================

  //turns uid into int
  def string_to_int(uid_string: String): BigInt = {
    val hex = uid_string.getBytes("utf8").map("%02x".format(_)).mkString
    BigInt(hex, 16)
  }

  //outputs hash values 
  def myhashs(uid_string: String): Seq[Long] = {

    //same as task1,turn to string but hash differently
    val x = string_to_int(uid_string)
    val hashes = hash_params.map { case (a, b) =>
      val hash_value = (BigInt(a) * x + BigInt(b)) mod BigInt(prime_number)
      hash_value.toLong
    }
    hashes.toSeq
  }

  def count_trailing_zeros(integer_in: Long): Int = {
    var integer = integer_in
    var count = 0

    //error handling 
    if (integer == 0L) {
      return 32
    }

    //while lsb is 0, keep going     
    while ((integer & 1L) == 0L) {
      count += 1

      //shifts by 1 bit to the right 
      integer = integer >>> 1
    }

    //count of zeros 
    count
  }

  def get_estimate(max_trailing_zeros_list: Array[Int]): Int = {

    // 2^R from flajolet martin 
    val estimates = max_trailing_zeros_list.map(r => math.pow(2.0, r.toDouble))

    //store averages of each group 
    val group_avgs = new Array[Double](num_groups)
    var g = 0
    while (g < num_groups) {
      val start = g * group_size
      val end = start + group_size
      //estimate for each number in group
      val group_slice = estimates.slice(start, end)

      //avg of each group
      val group_avg = group_slice.sum / group_size.toDouble

      //append to group avgs 
      group_avgs(g) = group_avg
      g += 1
    }

    //median to reduce variance from ends 
    val sorted = group_avgs.sorted

    //for median, depends on if even or odd to calc median
    val estimation =
      if (num_groups % 2 == 1) {
        sorted(num_groups / 2)
      } else {
        val mid = num_groups / 2
        (sorted(mid - 1) + sorted(mid)) / 2.0
      }

    // final integer estimate
    estimation.round.toInt
  }

  //process strema of users with the funcs built
  def process_stream(users: Array[String]): (Int, Int) = {

    //get unique users, then count for ground truth actual user counts
    val unique_users = users.toSet
    val ground_truth = unique_users.size

    //max is hash functions * array of [0]
    val max_trailing_zeros = Array.fill[Int](num_hash_functions)(0)

    //iterate through each uid to hash
    unique_users.foreach { uid =>
      val hashes = myhashs(uid)

      hashes.zipWithIndex.foreach { case (hash_value, i) =>
        //check trailing zeros
        val zeros = count_trailing_zeros(hash_value)

        //set new max trailing zeros if exceeds the current amount 
        if (zeros > max_trailing_zeros(i)) {
          max_trailing_zeros(i) = zeros
        }
      }
    }

    //use max to estimate vs ground truth (FM algo)
    val estimation = get_estimate(max_trailing_zeros)

    (ground_truth, estimation)
  }

  // =========================
  // main sawce
  // =========================

  def main(args: Array[String]): Unit = {
    //same as the usual
    val (input_filename, stream_size, num_of_asks, output_filename) = check_inputs(args)
    val bx = Blackbox()

    val results = mutable.ArrayBuffer[(Int, Int, Int)]()

    for (ask <- 0 until num_of_asks) {
      val batch = bx.ask(input_filename, stream_size)
      val (ground_truth, estimation) = process_stream(batch)
      results.append((ask, ground_truth, estimation))
    }

    write_output(output_filename, results.toSeq)
  }
}
