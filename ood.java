class Vehicle
{
    //attributes
    protected String licencePlate;

    //methods
    public Vehicle(String licencePlate)
    {
        this.licencePlate = licencePlate;
    }

}


class Car extends Vehicle
{
    public car(String carId)
    {
        super(carId);
    }
}

class Truck extends Vehicile
{
    public truck(String truckId)
    {
        super(truckId);
    }

}


class Motorcycle extends Vehicle
{
    public motorBike(String motoId)
    {
        super(motoId);
    }

}


class ParkingSpot
{
    private int id;
    private int size;
    private double rate;

    public ParkingSpot(int id, int size, double rate)
    {
        this.id = id;
        this.size = size;
        this.rate = rate;
    }

}
