(define (domain foodora_domain)
  (:requirements :strips :typing :adl)
  
  (:types node restaurant customer biker)
  
  (:predicates (edge ?node ?node)
               (at-r ?restaurant ?node)
			   (at-c ?customer ?node)
			   (at-b ?biker ?node)
			   (rGotFoodFor ?restaurant ?customer)
			   (bGotFoodFor ?biker ?customer)
			   (notHaveFood ?biker)
			   (gotFood ?customer))

  (:action move
    :parameters (?b - biker ?from - node ?to - node)
    :precondition (and (at-b ?b ?from) (edge ?from ?to))
    :effect (and (not (at-b ?b ?from)) (at-b ?b ?to))
   )

  (:action pickUpFood
    :parameters (?b - biker ?r - restaurant ?c - customer ?n - node)
    :precondition (and (at-b ?b ?n) (at-r ?r ?n) (notHaveFood ?b) (rGotFoodFor ?r ?c))
    :effect (and (not (notHaveFood ?b)) (bGotFoodFor ?b ?c) (not (rGotFoodFor ?r ?c)))
   )  
   
  (:action deliverFood
    :parameters (?b - biker ?c - customer ?n - node)
    :precondition (and (at-b ?b ?n) (at-c ?c ?n) (bGotFoodFor ?b ?c))
    :effect (and (notHaveFood ?b) (not (bGotFoodFor ?b ?c)) (gotFood ?c))
   )
   
 
 )