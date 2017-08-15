      program gen_test_fortran_files
          implicit none

          open(100,file='correct_file', access='sequential',
     x     status='unknown',form='unformatted')

          write(100) 'This is a test.'
          write(100) 'All the records in this file are correct.'
          write(100) 'No errors here.'

          ! Use stream access to simulate faulty sequential access
          open(200,file='mismatched_indicators_file', access='stream')

          write(200) 5
          write(200) 'asdfg'
          write(200) 5
          write(200) 5
          write(200) 'gfdsa'
          write(200) 9

          open(300,file='missing_indicator_file', access='stream')

          write(300) 5
          write(300) 'asdfg'

          open(400,file='incomplete_indicator_file', access='stream')

          write(400) 5
          write(400) 'asdfg'
          write(400) 'ab' ! two bytes where there should be a four byte indicator

          open(500,file='not_enough_data_file', access='stream')

          write(500) 100
          write(500) 'asdfg'

          open(600,file='empty_record_file', access='stream')
          write(600) 0
          write(600) 0

          open(700,file='value_error_file', access='stream')
          write(700) 10
          write(700) 'abcdefghij'
          write(700) 10
      end program


