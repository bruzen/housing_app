from model.agents import Realtor, Bid, Allocation

def test_sell_homes():
    realtor = Realtor()

    # Create some bids
    bid1 = Bid("Bidder1", "Property1", 100)
    bid2 = Bid("Bidder2", "Property1", 150)
    bid3 = Bid("Bidder3", "Property2", 200)
    bid4 = Bid("Bidder4", "Property2", 180)

    # Add the bids to the realtor
    realtor.add_bid(bid1)
    realtor.add_bid(bid2)
    realtor.add_bid(bid3)
    realtor.add_bid(bid4)

    # Sell homes and get the allocations
    allocations = realtor.sell_homes()

    # Check the allocations
    assert len(allocations) == 2

    allocation1 = allocations[0]
    assert allocation1.property == "Property1"
    assert allocation1.successful_bidder == "Bidder2"
    assert allocation1.bid_price == 150
    assert allocation1.final_price == 100

    allocation2 = allocations[1]
    assert allocation2.property == "Property2"
    assert allocation2.successful_bidder == "Bidder3"
    assert allocation2.bid_price == 200
    assert allocation2.final_price == 180
